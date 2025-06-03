import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging


API_KEY = "fd895b77ece1cd582d9d2a40cc6d23f88008"
BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
def fetch_ids_batch(doi, attempt=1):
    """Fetch PMCIDS and PMIDS using a batch request with retries."""
    params = {
        'format': 'json',
        'ids': doi,#','.join(doi_batch),
        'api_key': API_KEY
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
                return fetch_ids_batch(doi, attempt + 1)
        elif response.status_code in {500, 502, 503, 504}:
            wait_time = BACKOFF_FACTOR * attempt
            logging.warning(f"Server error {response.status_code}! Retrying after {wait_time} seconds (attempt {attempt})...")
            time.sleep(wait_time)
            if attempt < MAX_RETRIES:
                return fetch_ids_batch(doi, attempt + 1)
        else:
            logging.error(f"Request failed with status {response.status_code}: {response.text}")

    except requests.RequestException as e:
        logging.error(f"Request exception: {e}")
        if attempt < MAX_RETRIES:
            wait_time = BACKOFF_FACTOR * attempt
            logging.warning(f"Retrying after {wait_time} seconds (attempt {attempt})...")
            time.sleep(wait_time)
            return fetch_ids_batch(doi, attempt + 1)

    # If all retries fail, return errors
    return [(doi, 'Error', 'Error') for doi in doi]

def check_open_access(doi):
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {
        "email": "recheinje@gmail.com"
    }
    link = {}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('is_oa'): 
            link[doi] = data.get('best_oa_location', {}).get('url')
            return ['Yes', link[doi]]
            #print(f"DOI: {doi} is Open Access")
            #print(f"OA URL: {data.get('best_oa_location', {}).get('url')}")
        else:
            print(f"DOI: {doi} is not Open Access")
            return('No')
    else:
        print(f"Error: {response.status_code} for DOI: {doi}")


