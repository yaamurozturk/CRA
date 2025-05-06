#!/usr/bin/env python
# coding: utf-8
import os
import re
import gc
import time
import random
import argparse
import requests
import pandas as pd
import string
import spacy
from tqdm import tqdm
import lxml.etree as ET
from pysbd import Segmenter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description ='Citation context extraction from PMC xml files.')

# Mutually exclusive groupe
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-x", "--xml_file", nargs = 1, help ='Input PMC xml file')
group.add_argument("-f", "--xml_folder",  help ='Folder containing PMC xml files')
args = parser.parse_args()

BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
key = "fd895b77ece1cd582d9d2a40cc6d23f88008" # NCBI key
bs = 50 # batch size
par = 5 # workers
target_doi = "10.1016/j.cub.2017.05.064"
target_title = "Animal Communication: When I'm Calling You, Will You Answer Too?"
missing_pmids = []

def find_section_titles(element):
    """Find both the nearest and top-level section titles."""
    nearest_title = None
    top_level_title = None
    top_level_sec = None

    while element is not None:
        parent = element.getparent()
        if parent is not None and parent.tag == "sec":
            title = parent.find("title")
            if title is not None and title.text:
                if nearest_title is None:  # First found section title
                    nearest_title = title.text.strip()
                top_level_sec = parent  # Keep track of the outermost <sec>
        element = parent  # Move up the tree

    # If a top-level <sec> is found, get its title
    if top_level_sec is not None:
        top_title_elem = top_level_sec.find("title")
        if top_title_elem is not None and top_title_elem.text:
            top_level_title = top_title_elem.text.strip()

    return nearest_title or "Unknown Section", top_level_title or "Unknown Section"

def fetch_dois_batch(pmids):
    batch_size = 20
    results = {}
    j = 0

    def request_dois(ids):
        params = {'format': 'json', 'ids': ','.join(ids), 'api_key': key}
        try:
            response = requests.get(BASE_URL, params=params, timeout=20)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limit exceeded on batch!")
            else:
                print(f"Failed request with status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        return None

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        print(f'Requesting batch {j}')
        data = request_dois(batch)

        if data:
            invalid_pmids = []
            for rec in data.get('records', []):
                #print(rec)
                if 'errmsg' in rec and 'invalid article id' in rec['errmsg'].lower():
                    invalid_pmids.append(f"PMC{rec.get('pmid', '')}")
                else:
                    pmid = rec.get('pmid')
                    doi = rec.get('doi', pmid)
                    if pmid:
                        results[pmid] = doi

            if invalid_pmids:
                print(f"Retrying {len(invalid_pmids)} invalid PMIDs as PMC IDs")
                retry_data = request_dois(invalid_pmids)
                if retry_data:
                    for rec in retry_data.get('records', []):
                        #print(rec)
                        pmcid = rec.get('pmcid', '').replace('PMC', '')
                        doi = rec.get('doi', pmcid)
                        print(pmcid, doi)
                        if pmcid:
                            results[pmcid] = doi
        j += 1

    return results

def split_sentences(text):
    protected_abbr = ["et al.","e.g.","i.e.","Fig.","Ref.","No.","cf.","ca.","vs.","Dr.","Prof.","Mr.","Mrs.","Ms.",
                      "Inc.","Ltd.","Co.","Eq.","Vol.","pp.","Ch.","eds.","ed.","al.","Jr.","Sr.","St.","mg.", "mL.", "cm.", "mm.", "kg.", "km."]
    for abbr in protected_abbr:
        text = text.replace(abbr, abbr.replace(".", "<DOT>"))
    text = re.sub(r'\[(?:[^\[\]]*?[.;][^\[\]]*?)\]', lambda m: m.group(0).replace('.', '<DOT>').replace(';', '<SEMI>'), text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.replace('<DOT>', '.').replace('<SEMI>', ';') for s in sentences]
    return sentences


def parse_components(ref_id):
    match = re.match(r"([a-zA-Z]*)(\d+)([a-z]?)", ref_id)
    return match.groups() if match else (None, None, None)


def expand_citation_range(start_id, end_id, ref_list):
    start_prefix, start_num, start_suffix = parse_components(start_id)
    end_prefix, end_num, end_suffix = parse_components(end_id)
    if start_prefix != end_prefix or not start_num or not end_num:
        return [start_id, end_id]

    try:
        start_num = int(start_num)
        end_num = int(end_num)
    except ValueError:
        return [start_id, end_id]

    if start_num > end_num or (end_num - start_num) > 25:
        return [start_id, end_id]

    expanded = []
    for ref in ref_list:
        prefix, num, suffix = parse_components(ref)
        if prefix == start_prefix and num and start_num <= int(num) <= end_num:
            expanded.append(ref)
    return sorted(expanded, key=lambda x: int(parse_components(x)[1]))


def extract_pmc_citations(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        article = next(root.iter("article"), None)
        if article is None:
            return pd.DataFrame()

        citing_doi = pmcid = None
        for article_id in article.findall(".//article-id"):
            if article_id.get("pub-id-type") == "doi":
                citing_doi = article_id.text
            if article_id.get("pub-id-type") == "pmc":
                pmcid = article_id.text

        ref_dict, ref_list = {}, []
        for ref in article.findall(".//ref"):
            ref_id = ref.get("id")
            if not ref_id:
                continue
            ref_list.append(ref_id)
            title_elem = ref.find(".//article-title")
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title"
            doi_elem = ref.find(".//pub-id[@pub-id-type='doi']")
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None
            pmid_elem = ref.find(".//pub-id[@pub-id-type='pmid']")
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None
            ref_dict[ref_id] = {
                "title": title,
                "doi": doi if doi else pmid if pmid else "No DOI",
                "pmid": pmid
            }

        data = []
        for paragraph in article.findall(".//p"):
            full_text = " ".join(paragraph.itertext()).strip()
            if not full_text:
                continue
            sentences = split_sentences(full_text)
            xrefs = [elem for elem in paragraph.iter() if elem.tag == "xref" and elem.get("ref-type") == "bibr"]
            i = 0
            while i < len(xrefs):
                rid = xrefs[i].get("rid")
                citation_text = xrefs[i].text or ''
                # Check for stacked range pattern: xref - xref
                if i+1 < len(xrefs):
                    mid_text = xrefs[i].tail or ''
                    if '-' in mid_text or '–' in mid_text:
                        end_rid = xrefs[i+1].get("rid")
                        expanded = expand_citation_range(rid, end_rid, ref_list)
                        for ref_id in expanded:
                            sentence = next((s for s in sentences if citation_text in s), full_text)
                            data.append(create_citation_row(ref_dict, ref_id, full_text, sentence, citing_doi, pmcid))
                        i += 2
                        continue

                # Check for stacked individual citations: <xref/><xref/><xref/>
                if ',' in (xrefs[i].tail or ''):
                    sentence = next((s for s in sentences if citation_text in s), full_text)
                    data.append(create_citation_row(ref_dict, rid, full_text, sentence, citing_doi, pmcid))
                    i += 1
                    continue

                # Single xref
                sentence = next((s for s in sentences if citation_text in s), full_text)
                data.append(create_citation_row(ref_dict, rid, full_text, sentence, citing_doi, pmcid))
                i += 1

        return pd.DataFrame(data, columns=["DOI", "Cited title", "Citation ID", "CC paragraph", "Citation_context", "Citing DOI", "PMCID"])

    except Exception as e:
        print(f"Error processing file {xml_file}: {str(e)}")
        return pd.DataFrame()


def create_citation_row(ref_dict, ref_id, text, sentence, citing_doi, pmcid):
    ref_info = ref_dict.get(ref_id, {"title": "No title", "doi": "No DOI"})
    return [
        ref_info["doi"],
        ref_info["title"],
        ref_id,
        text,
        sentence,
        citing_doi,
        pmcid
    ]

def process_files(directory):
    """
    Process PMC XML files in parallel using multiprocessing.
    Prints the number of citations extracted from each file.

    Args:
        directory (str): Path to folder containing .xml files.

    Returns:
        tuple: (Combined DataFrame of citations, List of missing PMIDs)
    """
    start = time.time()

    # Gather all XML files in the directory
    files = [entry.path for entry in os.scandir(directory) 
             if entry.is_file() and entry.name.endswith('.xml')]

    print(f"Found {len(files)} XML files to process...")

    all_dfs = []
    all_missing_pmids = []

    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 3)) as executor:
        futures = {executor.submit(extract_pmc_citations, file): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing XML files", unit='file'):
            file = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    filename = os.path.basename(file)
                    num_citations = len(df)
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        total_citations = len(combined_df)
        print(f"\nTotal: Extracted {total_citations} citations from {len(all_dfs)} files in {time.time() - start:.2f} seconds")
        return combined_df#, all_missing_pmids
    else:
        print("No citation data extracted from any file.")
        return pd.DataFrame()#, []
        
def is_only_range_indicator(text):
    """Check if text is nothing but a range indicator."""
    return text in ["-", "–", "−"] or re.match(r"^\s*[–−-]\s*$", text)

def safe_expand_range(start_id, end_id, ref_list):
    """
    Safely expand a citation range with strict checks.
    Returns the original IDs if expansion isn't appropriate.
    """
    def parse_components(ref_id):
        match = re.match(r"([a-zA-Z]*)(\d+)([a-z]?)", ref_id)
        if match:
            return match.groups()
        return None, None, None

    start_prefix, start_num, start_suffix = parse_components(start_id)
    end_prefix, end_num, end_suffix = parse_components(end_id)

    # Only expand if: Prefixes match, Both have numbers and Numbers are in proper order
    
    if (not start_prefix or not end_prefix or 
        start_prefix != end_prefix or
        not start_num or not end_num):
        return [start_id, end_id]

    try:
        start_num = int(start_num)
        end_num = int(end_num)
    except (ValueError, TypeError):
        return [start_id, end_id]

    if start_num > end_num:
        return [start_id, end_id]

    # Don't expand unreasonably large ranges
    if (end_num - start_num) > 20:
        return [start_id, end_id]

    # Collect references in range
    expanded = []
    for ref in ref_list:
        ref_prefix, ref_num, ref_suffix = parse_components(ref)
        if not ref_num:
            continue
            
        try:
            ref_num = int(ref_num)
        except (ValueError, TypeError):
            continue

        if (ref_prefix == start_prefix and 
            start_num <= ref_num <= end_num and
            (not start_suffix or not end_suffix or 
             (ref_suffix and start_suffix <= ref_suffix <= end_suffix))):
            expanded.append(ref)

    # Sort numerically
    expanded.sort(key=lambda x: (parse_components(x)[1], parse_components(x)[2] or ''))
    
    return expanded if expanded else [start_id, end_id]
 
 
def normalize_dashes(text):
    """Convert all dash-like characters to standard hyphens."""
    return re.sub(r'[‐‒–—―−]', '-', str(text).lower())
       
def match_min_phrase(cited_title, target_title):
    """Check if cited_title contains the first critical part of target_title."""
    if pd.isnull(cited_title) or pd.isnull(target_title):
        return False
    
    min_phrase = "Animal Communication: When I'm Calling You"
    
    # Normalize both strings
    normalized_cited = normalize_dashes(cited_title)
    normalized_phrase = normalize_dashes(min_phrase)
    
    return normalized_phrase in normalized_cited
#################### Extract citations from xml file #########################
if __name__ == "__main__":

    if args.xml_file:
        pmc_citations_df = extract_pmc_citations(args.xml_file[0])
    elif args.xml_folder:
        pmc_citations_df = process_files(args.xml_folder)

# Apply filtering
final = pmc_citations_df[
    (pmc_citations_df['DOI'] == target_doi) |
    (pmc_citations_df['Cited title'].apply(
        lambda x: match_min_phrase(x, target_title) if pd.notna(x) else False))]

print(final)
#print(pmc_citations_df)
final.to_csv('test_vikers.tsv', sep = '\t', index=False)

