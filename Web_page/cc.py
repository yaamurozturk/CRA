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

nlp = spacy.load("en_core_web_sm") 
BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

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

def expand_citation_range(start_id, end_id,ref_list):
    if start_id in ref_list and end_id in ref_list:
        start_idx = ref_list.index(start_id)
        end_idx = ref_list.index(end_id)
        
        if start_idx < end_idx:  
            return ref_list[start_idx:end_idx + 1]
    
    return [start_id, end_id] 

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

        # Prep for batch
        pmid_cache = {}
        pmid_list = []
        pmid_to_refid = {}
        ref_dict = {}
        ref_list = []

        refs = first_article.findall(".//ref")


        for i, ref in enumerate(refs):
            ref_id = ref.get("id")
            title_elem = ref.find(".//article-title")
            doi_elem = ref.find(".//pub-id[@pub-id-type='doi']")
            pmid_elem = ref.find(".//pub-id[@pub-id-type='pmid']")
            ref_list.append(ref_id)

            if doi_elem is not None and doi_elem.text:
                doi_value = doi_elem.text.strip()
            elif pmid_elem is not None and pmid_elem.text:
                pmid = pmid_elem.text.strip()
                pmid_list.append(pmid)
                pmid_to_refid[ref_id] = pmid
                doi_value = None  
            else:
                doi_value = 'No DOI'

            ref_dict[ref_id] = {
                "title": title_elem.text.strip() if title_elem is not None and title_elem.text else "No title",
                "doi": doi_value
            }

        if pmid_list:
            print(f"Fetching DOIs for PMIDs batch: {pmid_list}")
            pmid_cache = fetch_dois_batch(pmid_list)

            for r_id, pmid in pmid_to_refid.items():
                ref_dict[r_id]["doi"] = pmid_cache.get(pmid, 'Not open access DOI')
            
            
        data = []
        for paragraph in first_article.findall(".//p"):
            text = " ".join(paragraph.itertext()).strip()
            sentences = split_sentences(text)
            citation_matches = paragraph.findall(".//xref")
            nearest, top_level = find_section_titles(paragraph)
            
            sentence_index = 0
            
            i = 0
            while i < len(citation_matches):
                citation = citation_matches[i]
                citation_id = citation.get("rid")

                last_sentence_index = 0
                # Check if there's an en-dash after the citation
                if i + 1 < len(citation_matches):
                    next_citation = citation_matches[i + 1]
                    sibling_text = citation.tail.strip() if citation.tail else ""

                    if "–" in sibling_text: 
                        next_citation_id = next_citation.get("rid")
                        expanded_ids = expand_citation_range(citation_id, next_citation_id,ref_list)
                        i += 1  
                
                i += 1  # Moving on
                
                # Find the sentence that contains the citation
                sentence_index = next(
                    (i for i, s in enumerate(sentences) 
                     if citation.text and re.search(rf'\b{re.escape(citation.text.strip())}\b', s.text)),
                    None
                )
                #print(sentences)
                #print(len(sentences))
                expanded_ids = [citation_id]
                previous_next = " boink "
                if sentence_index is not None:
                    prev_index = max(0, sentence_index - 1)
                    next_index = min(len(sentences), sentence_index + 2)
                    previous_next = " ".join([sent.text for sent in sentences[prev_index:next_index]]).strip()
                    #print(previous_next)

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

