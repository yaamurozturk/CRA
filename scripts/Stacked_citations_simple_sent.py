#!/usr/bin/env python
# coding: utf-8
import os
import re
import time
import argparse
import requests
import pandas as pd
import string
from tqdm import tqdm
import lxml.etree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description ='Citation context and abstract extraction from PMC xml files.')

# Mutually exclusive groupe
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-x", "--xml_file", nargs = 1, help ='Input PMC xml file')
group.add_argument("-f", "--xml_folder",  help ='Folder containing PMC xml files')
parser.add_argument("-o", "--output", nargs = 1, default = "cc.tsv", help = "Output file name, default = cc.tsv")
parser.add_argument("-c", "--context", nargs = 1, default = "basic", help = "Basic (basic) or with cos similarity (cos) citation context, default = basic")
args = parser.parse_args()

BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
key = "fd895b77ece1cd582d9d2a40cc6d23f88008" # NCBI key
bs = 50 # batch size
par = 5 # workers
pmid = {} 
no_pmid = []
failed_dois = []
dois_without_abstract = []
target_doi = "10.1002/anie.201109089"
target_title = "Carbon-dot-based dual-emission nanohybrid produces a ratiometric fluorescent sensor for in vivo imaging of cellular copper ions"
missing_pmids = []
# # **Extract all citations in xml file:  1. Remove all citations that have no DOI  2. Save a df with unique DOI**


def split_sentences(text):
    protected_abbr = ["et al.","e.g.","i.e.","Fig.","Ref.","No.","cf.","ca.","vs.","Dr.","Prof.","Mr.","Mrs.","Ms."
    ,"Inc.","Ltd.","Co.","Eq.","Vol.","pp.","Ch.","eds.","ed.","al.","Jr.","Sr.","St.","mg.", "mL.", "cm.", "mm.", "kg.", "km.",]
    for abbr in protected_abbr:
        text = text.replace(abbr, abbr.replace(".", "<DOT>"))
    text = re.sub(r'\[(?:[^\[\]]*?[.;][^\[\]]*?)\]', lambda m: m.group(0).replace('.', '<DOT>').replace(';', '<SEMI>'), text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.replace('<DOT>', '.').replace('<SEMI>', ';') for s in sentences]
    return sentences

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
    """Fetch DOIs for a batch of PMIDs."""
    batch_size = 25
    results = {}
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        params = {'format': 'json', 'ids': ','.join(batch), 'api_key': key}
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for rec in data.get('records', []):
                    pmid = rec.get('pmid', None)
                    doi = rec.get('doi', 'Not open access DOI')
                    if pmid:
                        results[pmid] = doi
            elif response.status_code == 429:
                print("Rate limit exceeded on batch!")
        except requests.RequestException as e:
            print(f"Batch request failed: {e}")
    return results

def extract_pmc_citations(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        first_article = next(root.iter("article"), None)
        
        if first_article is None:
            return pd.DataFrame()

        # Get citing DOI (the article's own DOI)
        citing_doi = None
        for article_id in first_article.findall(".//article-id"):
            if article_id.get("pub-id-type") == "doi":
                citing_doi = article_id.text
                break

        # Build reference dictionary
        ref_dict = {}
        ref_list = []
        pmids_to_fetch = []  # Store PMIDs missing DOIs

        for ref in first_article.findall(".//ref"):
            ref_id = ref.get("id")
            if not ref_id:
                continue
                
            ref_list.append(ref_id)
            
            title_elem = ref.find(".//article-title")
            if title_elem is None:
                title_elem = ref.find(".//source")

            title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title"

            doi_elem = ref.find(".//pub-id[@pub-id-type='doi']")
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

            pmid_elem = ref.find(".//pub-id[@pub-id-type='pmid']")
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None
            
            # Store PMIDs for later DOI lookup
            if not doi and pmid:
                pmids_to_fetch.append(pmid)
                missing_pmids.append(pmid)
            ref_dict[ref_id] = {
                "title": title,
                "doi": doi if doi else f"PMID:{pmid}" if pmid else "No DOI",
                "pmid": pmid
            }
        
        # Fetch DOIs for PMIDs (batch request)
        if pmids_to_fetch:
            
            pmid_doi_map = fetch_dois_batch(pmids_to_fetch)
            # Update ref_dict with found DOIs
            for ref_id, data in ref_dict.items():
                if data["pmid"]  in pmid_doi_map:
                    data["doi"] = pmid_doi_map[data["pmid"]]
                else:
                    data["doi"] = data["pmid"]

        # Process citations
        data = []
        for paragraph in first_article.findall(".//p"):
            full_text = " ".join(paragraph.itertext()).strip()
            if not full_text:
                continue
                
            sentences = split_sentences(full_text)
            citations = paragraph.findall(".//xref[@ref-type='bibr']")

            i = 0
            while i < len(citations):
                citation = citations[i]
                rid = citation.get("rid")
                citation_text = citation.text.strip() if citation.text and citation.text.strip() else ""
                citation_tail = citation.tail.strip() if citation.tail and citation.tail.strip() else ""
                
                
                current_sentence = None #full_text  # default to full paragraph if we can't find sentence
                if sentences:
                    for sent in sentences:
                        sent_text = sent.text if hasattr(sent, 'text') else str(sent)
                        if rid and rid in sent_text:
                            current_sentence = sent_text.strip()
                            break
                        if citation_text and citation_text in sent_text:
                            current_sentence = sent_text.strip()
                            break

                # CASE 1: Simple citation
                if not citation_text:
                    if rid:
                        data.append(create_citation_row(ref_dict, rid, full_text, current_sentence, citing_doi))
                    i += 1
                    continue
                
                # CASE 2: Citation with text that might indicate a range
                if "-" in citation_text or "–" in citation_text:
                    parts = re.split(r"[,\s;]+", citation_text)
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                            
                        # Handle actual ranges
                        if "-" in part or "–" in part:
                            range_parts = re.split(r"[–-]", part)
                            if len(range_parts) == 2:
                                prefix = re.match(r"([a-zA-Z]+)", rid).group(1) if rid else ''
                                start = f"{prefix}{range_parts[0].strip()}"
                                end = f"{prefix}{range_parts[1].strip()}"
                                expanded = safe_expand_range(start, end, ref_list)
                                for expanded_id in expanded:
                                    data.append(create_citation_row(ref_dict, expanded_id, full_text, current_sentence, citing_doi))
                        else:
                            # Single reference in comma-separated list
                            prefix = re.match(r"([a-zA-Z]+)", rid).group(1) if rid else ''
                            ref_id = f"{prefix}{part}"
                            data.append(create_citation_row(ref_dict, ref_id, full_text, current_sentence, citing_doi))
                    i += 1
                    continue
                
                # CASE 3: xrefs separated by dash
                if (not citation_text and 
                    is_only_range_indicator(citation_tail) and 
                    i < len(citations) - 1):
                    next_rid = citations[i+1].get("rid")
                    expanded = safe_expand_range(rid, next_rid, ref_list)
                    for expanded_id in expanded:
                        data.append(create_citation_row(ref_dict, expanded_id, full_text, current_sentence, citing_doi))
                    i += 2  # Skip next citation as we've processed it
                    continue
                
                # DEFAULT CASE: Process as single citation
                if rid:
                    data.append(create_citation_row(ref_dict, rid, full_text, current_sentence, citing_doi))
                i += 1

        df = pd.DataFrame(data, columns=["DOI", "Cited title", "Citation ID", "CC paragraph", "Citation_context", "Citing DOI"])
        return df, missing_pmids

    except Exception as e:
        print(f"Error processing file {xml_file}: {str(e)}")
        return pd.DataFrame()

def create_citation_row(ref_dict, ref_id, text, current_sentence, citing_doi):
    """Create a row of citation data."""
    ref_info = ref_dict.get(ref_id, {"title": "No title", "doi": "No DOI"})
    return [
        ref_info["doi"],
        ref_info["title"],
        ref_id,
        text,
        current_sentence,
        citing_doi
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
                df, missing_pmids = future.result()
                if not df.empty:
                    filename = os.path.basename(file)
                    num_citations = len(df)
                    all_dfs.append(df)
                    all_missing_pmids.extend(missing_pmids)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        total_citations = len(combined_df)
        print(f"\nTotal: Extracted {total_citations} citations from {len(all_dfs)} files in {time.time() - start:.2f} seconds")
        return combined_df, all_missing_pmids
    else:
        print("No citation data extracted from any file.")
        return pd.DataFrame(), []
        
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
    
    min_phrase = "carbon-dot-based dual-emission nanohybrid produces a ratiometric"
    
    # Normalize both strings
    normalized_cited = normalize_dashes(cited_title)
    normalized_phrase = normalize_dashes(min_phrase)
    
    return normalized_phrase in normalized_cited
#################### Extract citations from xml file #########################
if __name__ == "__main__":
    if args.xml_file:
        pmc_citations_df, missing_pmids = extract_pmc_citations(args.xml_file[0])
    elif args.xml_folder:
        pmc_citations_df,missing_pmids = process_files(args.xml_folder)

# Apply filtering for papers citing the target paper
final = pmc_citations_df[
    (pmc_citations_df['DOI'] == target_doi) |
    (pmc_citations_df['DOI'] == "10.1002/ange.201109089") |
    (pmc_citations_df['Cited title'].apply(
        lambda x: match_min_phrase(x, target_title) if pd.notna(x) else False
    ))
]
"""
with open('pmids_to_check.csv', 'w') as f:
    f.write(f"PMID\n")
    for item in missing_pmids:
        f.write(f"{item}\n")"""
        
print(final)
print(pmc_citations_df)
final.to_csv('Citing/citing_cc.tsv', sep = '\t', index=False)
#pmc_citations_df = pmc_citations_df[pmc_citations_df['DOI'] != 'No DOI']
pmc_citations_df.to_csv('cit.tsv', sep = '\t', index=False)    

