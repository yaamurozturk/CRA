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

url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
key = ".." # NCBI key
mail = 'recheinje@gmail.com'
USER_AGENT = f"YourAppName/1.0 (mailto:{mail})"
HEADERS = {"User-Agent": USER_AGENT}
MAX_WORKERS = 5  # Number of parallel requests
CROSSREF_URL = "https://api.crossref.org/works/"
bs = 150 # batch size
par = 10 # workers
pmid = {} 
no_pmid = []
failed_dois = []
dois_without_abstract = []


# **Extract Abstract from PMC:   1. Using the pmids in the dict   2. Using epost for bulk queries**

def fetch_ids_batch(doi_batch):
    """ 
    This function fetches PMIDS using batch resquest to minimize the request number. Recalls itself
    when the rate limit is exceeded, retries after 10 seconds on the same batch that failed. 
    """
    no_id = []
    pid = {}
    params = {'format': 'json','ids': ','.join(doi_batch), 'api_key': key}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for rec in data.get('records', []):
                if 'pmid' in rec:
                    pid.update({rec['doi']: rec['pmid']})
                else : 
                    no_id.append(rec['doi'])
            return (pid, no_id)
        elif response.status_code == 429:
            print("Rate limit exceeded! Retrying after 10 seconds...")
            time.sleep(10)
            return fetch_ids_batch(doi_batch)  # Retry the same batch
    except requests.RequestException as e:
        pid.update({"No DOI": f"Request failed:{e}"})
        return pid, f"Request failed: {e}"
    pid.update({"No DOI":f"Response status code: {response.status_code}"})
    return pid, f"Response status code: {response.status_code}"
    
def process_batch(b):
    return fetch_ids_batch(b) 
    
def get_pubmed_abstracts_bulk(pmid_dict):
    """
    Fetch abstracts in bulk using epost + efetch while keeping DOI keys.
    Returns: 
        - dict {DOI: Abstract}
        - DataFrame with DOI, Abstract
    """
    # Extract PMIDs
    pmid_list = list(pmid_dict.values())

    # epost to store PMIDs up to 10 000 at once
    epost_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi"
    epost_params = {"db": "pubmed", "id": ",".join(pmid_list)}
    
    epost_response = requests.post(epost_url, data=epost_params, timeout=10)
    epost_response.raise_for_status()

    # Extract WebEnv & QueryKey using lxml
    root = ET.fromstring(epost_response.content)  # Use .content to get bytes instead of text
    webenv = root.find(".//WebEnv").text
    query_key = root.find(".//QueryKey").text

    # efetch to retrieve abstracts
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retmode": "xml",
        "rettype": "abstract"
    }
    
    efetch_response = requests.get(efetch_url, params=efetch_params, timeout=10)
    efetch_response.raise_for_status()
    
    # Parse XML to extract abstracts using lxml
    root = ET.fromstring(efetch_response.content)
    abstracts = {}

    for article in root.xpath(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        abstract_element = article.find(".//AbstractText")
        abstract_text = abstract_element.text if abstract_element is not None else "No abstract found"

        # Find corresponding DOI
        doi = next((k for k, v in pmid_dict.items() if v == pmid), None)
        if doi:
            abstracts[doi] = abstract_text

    # Convert to DataFrame
    df = pd.DataFrame(abstracts.items(), columns=["DOI", "Abstract"])
    
    return df
    
# # **Extract Abstract from Crossref:  1. Using the dois in the list**
c = 0

def get_crossref_metadata(doi):
    """Fetch metadata from CrossRef for a single DOI."""
    url = f"{CROSSREF_URL}{doi}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout = 25)

        if response.status_code == 200:
            data = response.json().get("message", {})
            abstract = data.get("abstract", "Not found")
            
            if abstract is None or abstract == "Not found":
              dois_without_abstract.append(doi)
              global c 
              c += 1
            return {
                "DOI": doi,
               # "Publisher": data.get("publisher", "Unknown Publisher"),
                #"Journal": data.get("container-title", ["Unknown Journal"])[0],
                "Abstract": abstract,
               # "Journal Article": "Yes" if data.get("type") == "journal-article" else "No",
            }

        elif response.status_code == 404:
            return {"DOI": doi, "Abstract": "Not found"}

        else:
            failed_dois.append(doi)  # Save DOI for retry
            return {"DOI": doi, "Abstract": "Not found"}

    except requests.RequestException as e:
        failed_dois.append(doi)  # Save DOI for retry
        return {"DOI": doi, "Abstract": "Not found"}

def fetch_metadata_parallel(doi_list):
    """Fetch metadata for a list of DOIs in parallel using ThreadPoolExecutor."""
    metadata_list = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_crossref_metadata, doi): doi for doi in doi_list}

        for future in tqdm(as_completed(futures), total=len(doi_list), desc="Fetching CrossRef abstracts", unit="DOI"):
            metadata_list.append(future.result())

    return metadata_list

def merge(dfs, abstracts_df,metadata_df , pmc_citations_df):
	for d in dfs :
		if not d.empty:
			d = d.drop_duplicates(subset=['DOI'])

	# Merge the initial df with duplicates DOIs with the result abstracts from PMC, CR and EL 
	merged_df = pmc_citations_df.merge(abstracts_df, on="DOI", how="left") \
		                    .merge(metadata_df, on="DOI", how="left", suffixes=("", "_cr")) 
	col = "Abstract" 
	# Retreiving the column with abstract 
	for c in [f"{col}", f"{col}_cr"]: #, f"{col}_el"]: 
	    pmc_citations_df[c] = pmc_citations_df["DOI"].map(abstracts_df.set_index("DOI")[col])

	# Drop redundant columns after selection
	columns_to_drop = [f"{col}_cr"]# +\
			  #[f"{col}_el"]

	merged_df.drop(columns=columns_to_drop, inplace=True)
	print(merged_df)
	print("merged length ", len(merged_df))
	merged_df = merged_df.dropna(subset = [col]) # remove citations where the abstract was not found
	return merged_df
    
"""############# Abstarct Extraction ###############"""    
def abstract(unique_dois,pmc_citations_df):
    batches = [unique_dois[i:i + bs] for i in range(0, len(unique_dois), bs)]
    with ThreadPoolExecutor(max_workers = par) as executor:
        results = list(executor.map(process_batch, batches))

    for pmid_dict, no_pmid_list in results:
        #print(pmid_dict)
        pmid.update(pmid_dict) 
        no_pmid.extend(no_pmid_list) 

################  Fetching abstracts from PMC and  Crossref ################
    # Check PMC 
    if len(pmid) > 0:
        abstracts_df = get_pubmed_abstracts_bulk(pmid)
        p = len(abstracts_df.dropna())
        print(f"Completed fetching {p} abstracts from PMC.")  

    # Check CrossRef
    if len(no_pmid) > 0:
        metadata_results = fetch_metadata_parallel(no_pmid)
        metadata_df = pd.DataFrame(metadata_results)

        print(f"Completed fetching {len(metadata_df) - c} abstracts from CrossRef.")
        if failed_dois:
          with open("failed_dois.txt", "w") as f:
              for doi in failed_dois:
                f.write(doi + "\n")
          print(f"{len(failed_dois)} requests failed. Saved to failed_dois.txt for retry.")   

            
    dfs = [abstracts_df, metadata_df]
    final_df = merge(dfs, abstracts_df,metadata_df ,pmc_citations_df)
    print(final_df)
    final_df.to_csv('athan.tsv', sep = '\t', index=False)
    return(final_df)
