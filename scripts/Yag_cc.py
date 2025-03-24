#!/usr/bin/env python
# coding: utf-8
import os
import re
import time
import json
import random
import spacy 
import argparse
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import lxml.etree as ET
from itertools import islice
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description ='Citation context and abstract extraction from PMC xml files.')

# Mutually exclusive groupe
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-x", "--xml_file", nargs = 1, help ='Input PMC xml file')
group.add_argument("-f", "--xml_folder",  help ='Folder containing PMC xml files')
parser.add_argument("-o", "--output", nargs = 1, default = "cc.tsv", help = "Output file name, default = cc.tsv")
parser.add_argument("-c", "--context", nargs = 1, default = "basic", help = "Basic (basic) or with cos similarity (cos) citation context, default = basic")
args = parser.parse_args()

nlp = spacy.load("en_core_web_sm") 
BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
key = ".." # NCBI key
bs = 150 # batch size
par = 10 # workers
pmid = {} 
no_pmid = []
failed_dois = []
dois_without_abstract = []
target_doi = "10.1002/anie.201109089"
target_title = "Carbon-dot-based dual-emission nanohybrid produces a ratiometric fluorescent sensor for in vivo imaging of cellular copper ions"

# # **Extract all citations in xml file:  1. Remove all citations that have no DOI  2. Save a df with unique DOI**

def split_sentences(text):
    """Splits text into sentences using spacy"""
    doc = nlp(text)  
    #print(doc)
    lis = []
    for sent in doc.sents: 
        lis.append(sent)
    return lis

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
    batch_size = 50
    results = {}
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        params = {'format': 'json', 'ids': ','.join(batch)}
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

def expand_citation_range(start_id, end_id,ref_list):
    if start_id in ref_list and end_id in ref_list:
        start_idx = ref_list.index(start_id)
        end_idx = ref_list.index(end_id)
        
        if start_idx < end_idx:  
            return ref_list[start_idx:end_idx + 1]
    
    return [start_id, end_id] 
    
import re
import string
import xml.etree.ElementTree as ET
import pandas as pd

def expand_citation_range(start_id, end_id, ref_list):
    def parse_rid(rid):
        match = re.match(r"([a-zA-Z]*)(\d+)([a-z]?)", rid)
        if match:
            prefix, num, suffix = match.groups()
            return prefix, int(num), suffix
        return None, None, None

    start_prefix, start_num, start_suffix = parse_rid(start_id)
    end_prefix, end_num, end_suffix = parse_rid(end_id)

    if start_prefix != end_prefix or start_num is None or end_num is None:
        return [start_id, end_id]  # fallback

    expanded = []
    if start_num == end_num and start_suffix and end_suffix:
        # e.g., B5a–B5c
        suffix_range = string.ascii_lowercase[string.ascii_lowercase.index(start_suffix):string.ascii_lowercase.index(end_suffix) + 1]
        for suffix in suffix_range:
            candidate = f"{start_prefix}{start_num}{suffix}"
            if candidate in ref_list:
                expanded.append(candidate)
    else:
        # e.g., B5–B9
        for n in range(start_num, end_num + 1):
            candidates = [f"{start_prefix}{n}"]
            candidates += [r for r in ref_list if r.startswith(f"{start_prefix}{n}") and r != f"{start_prefix}{n}"]
            for c in candidates:
                if c in ref_list:
                    expanded.append(c)
    return expanded or [start_id, end_id]

def parse_inline_citation(text, rid, ref_list):
    citation_ids = []
    parts = re.split(r"[,;]", text)
    for part in parts:
        part = part.strip()
        if re.search(r"[–-]", part):
            nums = re.split(r"[–-]", part)
            if len(nums) == 2:
                prefix = re.match(r"([a-zA-Z]+)", rid).group(1) if rid else ''
                start_id = f"{prefix}{nums[0].strip()}"
                end_id = f"{prefix}{nums[1].strip()}"
                citation_ids += expand_citation_range(start_id, end_id, ref_list)
        else:
            prefix = re.match(r"([a-zA-Z]+)", rid).group(1) if rid else ''
            citation_ids.append(f"{prefix}{part.strip()}")
    return citation_ids

def extract_pmc_citations(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        citing_doi = None
        first_article = next(root.iter("article"), None)
        if first_article is None:
            print("No <article> found in the XML file.")
            return pd.DataFrame()

        for article_id in first_article.findall(".//article-id"):
            if article_id.get("pub-id-type") == "doi":
                citing_doi = article_id.text
                break

        ref_dict = {}
        ref_list = []
        refs = first_article.findall(".//ref")

        for ref in refs:
            ref_id = ref.get("id")
            ref_list.append(ref_id)
            title_elem = ref.find(".//article-title")
            doi_elem = ref.find(".//pub-id[@pub-id-type='doi']")
            pmid_elem = ref.find(".//pub-id[@pub-id-type='pmid']")
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else "No DOI"
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None
            ref_dict[ref_id] = {
                "title": title_elem.text.strip() if title_elem is not None else "No title",
                "doi": doi if doi else f"PMID:{pmid}" if pmid else "No DOI"
            }

        data = []
        for paragraph in first_article.findall(".//p"):
            text = " ".join(paragraph.itertext()).strip()
            citation_matches = paragraph.findall(".//xref")

            i = 0
            while i < len(citation_matches):
                citation = citation_matches[i]
                citation_id = citation.get("rid")
                tail = citation.tail.strip() if citation.tail else ""
                expanded_ids = []

                # Case 1: single <xref> with inline 5,7 or 5–7 inside
                if citation.text and (',' in citation.text or re.search(r"\d+[–-]\d+", citation.text)):
                    expanded_ids = parse_inline_citation(citation.text, citation_id, ref_list)

                # Case 2: stacked <xref> like <xref>5</xref> – <xref>7</xref>
                elif re.search(r"[–-]", tail) and i + 1 < len(citation_matches):
                    next_citation = citation_matches[i + 1]
                    next_id = next_citation.get("rid")
                    expanded_ids = expand_citation_range(citation_id, next_id, ref_list)
                    i += 1

                # Case 3: normal <xref> single citation
                else:
                    expanded_ids.append(citation_id)

                for expanded_id in expanded_ids:
                    if expanded_id in ref_dict:
                        data.append([
                            citing_doi,
                            expanded_id,
                            text,
                            ref_dict[expanded_id]["title"],
                            ref_dict[expanded_id]["doi"]
                        ])
                i += 1

        df = pd.DataFrame(data, columns=["Citing DOI", "Citation ID", "CC paragraph", "Cited title", "DOI"])
        return df

    except Exception as e:
        print(f"Error processing file: {e}")
        return pd.DataFrame()




    
def process_files(directory):
    """Process XML files in parallel using multiprocessing."""
    start = time.time()
    files = [entry.path for entry in os.scandir(directory) if entry.is_file() and entry.name.endswith('.xml')]

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(extract_pmc_citations, file): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing XML files", unit='file', leave=True, position=0):
            result = future.result()
            if not result.empty:
                results.append(result)


    df = pd.concat(results, ignore_index=True)  # Concatenate all results at once
    print(f"Citations extracted in {time.time() - start:.2f} seconds")
    #print("Columns: ", df.columns) 
    return df
        
start_time = time.time()

############################################## Extract citations from xml file ######################################
if (args.xml_file):
	pmc_citations_df = extract_pmc_citations(args.xml_file[0])
elif (args.xml_folder):
	pmc_citations_df = process_files(args.xml_folder)
	
######################################### Simple paragraph citation context #####################################

def match_first_5_words(cited_title, target_title):
    if not cited_title or pd.isnull(cited_title) or pd.isnull(target_title):
        return False
    cited_words = cited_title.lower().split()[:5]  # Get the first 5 words
    target_title_lower = target_title.lower()
    # Check if all first 5 words appear in the target_title
    return all(word in target_title_lower for word in cited_words)

final = pmc_citations_df[
    (pmc_citations_df['DOI'] == target_doi) | 
    (pmc_citations_df['Cited title'].apply(lambda x: match_first_5_words(x, target_title)))
]


#final = pmc_citations_df[(pmc_citations_df['DOI'] == target_doi) | (pmc_citations_df['Cited title'] == target_title)]
print(final)
print(pmc_citations_df)
final.to_csv('citing_citations.tsv', sep = '\t', index=False)
#pmc_citations_df = pmc_citations_df[pmc_citations_df['DOI'] != 'No DOI']
pmc_citations_df.to_csv('all_citations.tsv', sep = '\t', index=False)    
