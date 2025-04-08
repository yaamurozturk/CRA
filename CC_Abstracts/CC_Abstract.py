import os
import re
import time
import json
import random
import spacy 
import shutil
import argparse
import requests
import numpy as np
import pandas as pd
import string
from tqdm import tqdm
import lxml.etree as ET
from itertools import islice
from functions_abstract import *
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
key = ".." # NCBI key
path = "../test"
bs = 150 # batch size
par = 10 # workers
pmid = {} 
no_pmid = []
failed_dois = []
dois_without_abstract = []

"""###################### Citation Context Extraction ###########################""" 
def get_nlp():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp
def split_sentences(text):
    if not hasattr(split_sentences, "nlp"):
        split_sentences.nlp = get_nlp()

    doc = split_sentences.nlp(text)
    return list(doc.sents)
    
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
            
            # Extract metadata
            title_elem = ref.find(".//article-title") or ref.find(".//source")
            title = title_elem.text.strip() if title_elem is not None else "No title"
            
            doi_elem = ref.find(".//pub-id[@pub-id-type='doi']")
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None
            
            pmid_elem = ref.find(".//pub-id[@pub-id-type='pmid']")
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None
            
            # Store PMIDs for later DOI lookup
            if not doi and pmid:
                pmids_to_fetch.append(pmid)
            
            ref_dict[ref_id] = {
                "title": title,
                "doi": doi if doi else f"PMID:{pmid}" if pmid else "No DOI",
                "pmid": pmid
            }

        # Fetch DOIs for PMIDs (batch request)
        if pmids_to_fetch:
            pmid_doi_map = fetch_dois_batch(pmids_to_fetch)  # Uses your existing function
            # Update ref_dict with found DOIs
            for ref_id, data in ref_dict.items():
                if data["pmid"] and data["pmid"] in pmid_doi_map:
                    data["doi"] = pmid_doi_map[data["pmid"]]
        # Process citations
        data = []
        for paragraph in first_article.findall(".//p"):
            text = " ".join(paragraph.itertext()).strip()
            sentences = split_sentences(text)
            citations = paragraph.findall(".//xref[@ref-type='bibr']")

            
            i = 0
            while i < len(citations):
                citation = citations[i]
                rid = citation.get("rid")
                citation_text = citation.text.strip() if citation.text else ""
                citation_tail = citation.tail.strip() if citation.tail else ""
                
                 # Find containing sentence
                sentence_index = None
                for idx, sent in enumerate(sentences):
                    if citation_text and citation_text in sent.text:  # Ensure we are comparing text, not token
                        sentence_index = idx
                        break

                previous_next = ""
                if sentence_index is not None:
                    # Handle sentence bounds correctly
                    prev_idx = max(0, sentence_index - 1)
                    next_idx = min(len(sentences), sentence_index + 2)
                    previous_next = " ".join([s.text for s in sentences[prev_idx:next_idx]]).strip()  # Get the text of sentences
                    #print(f"Previous and next context: {previous_next}")
                else:
                    print(f"Citation text '{citation_text}' not found in any sentence.")

                # CASE 1: Simple citation
                if not citation_text:
                    if rid:  # Only process if we have a reference ID
                        data.append(create_citation_row(ref_dict, rid, text, previous_next, top_level, nearest, citing_doi))
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
                                    data.append(create_citation_row(ref_dict, rid, text, previous_next, citing_doi))
                        else:
                            # Single reference in comma-separated list
                            prefix = re.match(r"([a-zA-Z]+)", rid).group(1) if rid else ''
                            ref_id = f"{prefix}{part}"
                            data.append(create_citation_row(ref_dict, rid, text, previous_next, citing_doi))
                    i += 1
                    continue
                
                # CASE 3: xrefs separated by dash
                if (not citation_text and 
                    is_only_range_indicator(citation_tail) and 
                    i < len(citations) - 1):
                    next_rid = citations[i+1].get("rid")
                    expanded = safe_expand_range(rid, next_rid, ref_list)
                    for expanded_id in expanded:
                        data.append(create_citation_row(ref_dict, rid, text, previous_next, citing_doi))
                    i += 2  # Skip next citation as we've processed it
                    continue
                
                # DEFAULT CASE: Process as single citation
                if rid:
                    data.append(create_citation_row(ref_dict, rid, text, previous_next, citing_doi))
                i += 1

        df = pd.DataFrame(data, columns=["DOI", "Cited title", "Citation ID", "CC paragraph", "CC window", "Citing DOI"])
        return df

    except Exception as e:
        print(f"Error processing file: {e}")
        return pd.DataFrame()

def create_citation_row(ref_dict, ref_id, text, previous_next, citing_doi):
    """Create a row of citation data."""
    ref_info = ref_dict.get(ref_id, {"title": "No title", "doi": "No DOI"})
    print(ref_info["doi"])
    return [
        ref_info["doi"],
        ref_info["title"],
        ref_id,
        text,
        previous_next,
        citing_doi
    ]


def process_files(directory):
    """
    Process PMC XML files in parallel using multiprocessing.

    Args:
        directory (str): Path to folder containing .xml files.

    Returns:
        pd.DataFrame: Combined DataFrame with extracted citation data.
    """
    start = time.time()

    # Gather all XML files in the directory
    files = [entry.path for entry in os.scandir(directory) 
             if entry.is_file() and entry.name.endswith('.xml')]

    print(f"Found {len(files)} XML files to process...")

    results = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()-2) as executor:
        # Submit all jobs
        futures = {executor.submit(extract_pmc_citations, file): file for file in files}

        # Collect results as they finish
        for future in tqdm(as_completed(futures), 
                           total=len(futures), 
                           desc="Processing XML files", 
                           unit='file', 
                           leave=True, 
                           position=0):
            file = futures[future]
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    if results:
        df = pd.concat(results, ignore_index=True)
        print(f"Citations extracted from {len(results)} files in {time.time() - start:.2f} seconds")
        return df
    else:
        print("No citation data extracted from any file.")
        return pd.DataFrame()  
        


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
    
"""############ Fetch PMIDS ##################"""

def fetch_ids_batch(doi_batch, attempt=1):
    """Fetch PMCIDS and PMIDS using a batch request with retries."""
    params = {
        'format': 'json',
        'ids': ','.join(doi_batch),
        'api_key': key
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [(rec.get('doi', 'Not Found'), rec.get('pmid', 'Not Found'), rec.get('pmcid', 'Not Found'))
                    for rec in data.get('records', [])]
        elif response.status_code == 429:
            wait_time = BACKOFF_FACTOR * attempt
            logging.warning(f"Rate limit exceeded! Retrying after {wait_time} seconds (attempt {attempt})...")
            time.sleep(wait_time)
            if attempt < MAX_RETRIES:
                return fetch_ids_batch(doi_batch, attempt + 1)
        elif response.status_code in {500, 502, 503, 504}:
            wait_time = BACKOFF_FACTOR * attempt
            logging.warning(f"Server error {response.status_code}! Retrying after {wait_time} seconds (attempt {attempt})...")
            time.sleep(wait_time)
            if attempt < MAX_RETRIES:
                return fetch_ids_batch(doi_batch, attempt + 1)
        else:
            logging.error(f"Request failed with status {response.status_code}: {response.text}")

    except requests.RequestException as e:
        logging.error(f"Request exception: {e}")
        if attempt < MAX_RETRIES:
            wait_time = BACKOFF_FACTOR * attempt
            logging.warning(f"Retrying after {wait_time} seconds (attempt {attempt})...")
            time.sleep(wait_time)
            return fetch_ids_batch(doi_batch, attempt + 1)

    # If all retries fail, return errors
    return [(doi, 'Error', 'Error') for doi in doi_batch]


def doi_to_pmcid(dois):
    batch_size = 150  # Number of DOIs per request
    nb_workers = 5  # Number of parallel requests
    results = []

    # Split DOIs into batches of 10
    doi_batches = [dois[i:i + batch_size] for i in range(0, len(dois), batch_size)]

    # Execute requests in parallel using ThreadPoolExecutor with 5 workers
    with ThreadPoolExecutor(max_workers = nb_workers) as executor:
        results_list = list(executor.map(fetch_ids_batch, doi_batches)) # This returns a list of lists (batches) of tuples

    # Flatten results (since each batch returns a list)
    results = [item for sublist in results_list for item in sublist] # This is a list of tuples after flattening the batches lists

    # Convert results to DataFrame and save as CSV
    result_df = pd.DataFrame(results, columns=['DOI', 'PMID', 'PMCID'])
    result_df.to_csv('doi_to_pmcids.tsv', index=False, sep='\t')

    # Record end time
    end = time.time()

    print(f"Completed in {round(end-start, 2)} seconds.")
    return result_df

"""############### Download PubMed XML files #########""" 

def fetch_pubmed_articles(pmc_ids, batch_size=200):
    """Fetch articles in batches from PMC using E-utilities."""
    if os.path.exists(path):  
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    epost_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    for i in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} with {len(batch)} IDs...")
        
        # Post IDs to Entrez
        epost_params = {"db": "pmc", "id": ",".join(batch),"api_key": key}
        epost_response = requests.post(epost_url, data=epost_params, timeout=10)
        epost_response.raise_for_status()
        
        # Parse WebEnv and QueryKey
        root = ET.fromstring(epost_response.content)
        webenv = root.find(".//WebEnv").text
        query_key = root.find(".//QueryKey").text
        print("Received WebEnv and QueryKey.")

        # Fetch articles
        efetch_params = {
            "db": "pmc",
            "query_key": query_key,
            "WebEnv": webenv,
            "retmode": "xml",
            "rettype": "full",
            "api_key": key
        }
        efetch_response = requests.get(efetch_url, params=efetch_params, timeout=30)
        efetch_response.raise_for_status()
        
        # Parse and save articles
        root = ET.fromstring(efetch_response.content)
        for article in root.xpath(".//article"):
            pmid = article.find(".//article-meta/article-id[@pub-id-type='pmc']")
            pmid_text = pmid.text if pmid is not None else "unknown"

            file_path = os.path.join(path, f"{pmid_text}.xml")
            with open(file_path, "wb") as f:
                f.write(ET.tostring(article, encoding="utf-8", pretty_print=True))
            print(f"Downloaded: {pmid_text}.xml") 


             
"""########## Main Function #############"""

if __name__ == "__main__":
    df = pd.read_csv('../data/metadata.csv', dtype=str)
    dois = df['DOI'].dropna().unique()
    start = time.time()
    ids = doi_to_pmcid(dois)
    pmcid = ids[ids.PMCID != 'Not Found']
    p = pmcid.PMCID.head(5)
    fetch_pubmed_articles(p)
    pmc_citations_df = process_files(path)
    print(pmc_citations_df)
    unique_dois = pmc_citations_df["DOI"].unique()
    abstract(unique_dois,pmc_citations_df)
    
    #print(pmc_citations_df)
    
