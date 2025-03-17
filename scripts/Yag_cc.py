#!/usr/bin/env python
# coding: utf-8
import os
import re
import time
import json
import random
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


url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
key = "fd895b77ece1cd582d9d2a40cc6d23f88008" # NCBI key
bs = 150 # batch size
par = 10 # workers
pmid = {} 
no_pmid = []
failed_dois = []
dois_without_abstract = []

# # **Extract all citations in xml file:  1. Remove all citations that have no DOI  2. Save a df with unique DOI**

def split_sentences(text):
    """Splits text into sentences while preserving structure."""
    return re.split(r'(?<=[.!?])\s+', text)

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

def expand_citation_range(start_id, end_id):
    if start_id in ref_list and end_id in ref_list:
        start_idx = ref_list.index(start_id)
        end_idx = ref_list.index(end_id)
        
        if start_idx < end_idx:  
            return ref_list[start_idx:end_idx + 1]
    
    return [start_id, end_id] 

def extract_pmc_citations(xml_file):
    """Extract citations from a PMC XML file, ensuring citation ranges are handled correctly."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        citing_doi = None
        first_article = next(root.iter("article"), None)  # Get the first <article> tag and ignore all else
        if first_article is None:
            print("No <article> found in the XML file.")
            return pd.DataFrame()

        for article_id in first_article.findall(".//article-id"):
            if article_id.get("pub-id-type") == "doi":
                citing_doi = article_id.text
                break
        
        ref_dict = {}
        ref_list = []
        for ref in first_article.findall(".//ref"):
            ref_id = ref.get("id")
            title_elem = ref.find(".//article-title")
            doi_elem = ref.find(".//pub-id[@pub-id-type='doi']")
            ref_list.append(ref_id)
            ref_dict[ref_id] = {
                "title": title_elem.text.strip() if title_elem is not None and title_elem.text else "No title",
                "doi": doi_elem.text.strip() if doi_elem is not None and doi_elem.text else 'No DOI',
            }

        data = []
        for paragraph in first_article.findall(".//p"):
            text = " ".join(paragraph.itertext()).strip()
            sentences = split_sentences(text)
            citation_matches = paragraph.findall(".//xref")
            nearest, top_level = find_section_titles(paragraph)

            i = 0
            while i < len(citation_matches):
                citation = citation_matches[i]
                citation_id = citation.get("rid")

                expanded_ids = [citation_id]
                
                # Check if there's an en-dash after the citation
                if i + 1 < len(citation_matches):
                    next_citation = citation_matches[i + 1]
                    sibling_text = citation.tail.strip() if citation.tail else ""

                    if "–" in sibling_text:  # En-dash detected, meaning a range exists
                        next_citation_id = next_citation.get("rid")
                        expanded_ids = expand_citation_range(citation_id, next_citation_id,ref_list)
                        i += 1  # Skip the next citation as it's already included in the range
                
                i += 1  # Moving on

                # Find the sentence that contains the citation
                sentence_index = next(
                    (idx for idx, s in enumerate(sentences) if citation.text and re.search(rf'\b{re.escape(citation.text.strip())}\b', s)),
                    None
                )
                
                if sentence_index is not None:
                    prev_index = max(0, sentence_index - 1)
                    next_index = min(len(sentences), sentence_index + 2)
                    previous_next = " ".join(sentences[prev_index:next_index]).strip()

                for expanded_id in expanded_ids:
                    if expanded_id in ref_dict:
                        data.append([
                            citing_doi,
                            expanded_id,
                            text,  
                            previous_next, 
                            top_level,
                            nearest,
                            ref_dict[expanded_id]["title"],
                            ref_dict[expanded_id]["doi"]
                        ])

        df = pd.DataFrame(data, columns=["Citing DOI", "Citation ID", "CC paragraph", "CC prev_next", "IMRD", "Section title", "Cited title", "DOI"])
        return df[df['DOI'] != 'No DOI']

    except Exception as e:
        print(f"Error processing file: {e}")
        return pd.DataFrame()

        
def process_files(directory):
    """Process XML files in parallel using multiprocessing."""
    start = time.time()
    files = [entry.path for entry in os.scandir(directory) if entry.is_file() and entry.name.endswith('.xml')]

    results = []
    with ProcessPoolExecutor(max_workers = os.cpu_count()) as executor: 
        futures = {executor.submit(extract_pmc_citations, file): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing XML files", unit='file'):
            results.append(future.result())

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
print(pmc_citations_df)
pmc_citations_df.to_csv('out', sep = '\t', index=False)



