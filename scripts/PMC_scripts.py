import os
import time
import shutil
import requests
import argparse
import pandas as pd
import lxml.etree as ET
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description ='Citation context and abstract extraction from PMC xml files.')

parser.add_argument("-f", "--convert_file", nargs = 1,  help ='Convert DOIs into PMCIDs/PMIDs from file')
parser.add_argument("-c", "--convert", nargs = '+' , help ='Convert DOIs into PMCIDs/PMIDs from given list')
parser.add_argument("-n", "--file", nargs = 1, default = "dois_to_pmcids.tsv", help = "File name, default = dois_to_pmcids.tsv")
parser.add_argument("-o", "--output", nargs = 1, default = "pmc_xml", help = "Folder name, default = pmc_xml")
parser.add_argument("-d", "--download", nargs = 1, default = 'n', help ='Download the xml files of the DOIs having a PMCID (y/n), default: no (n)')
parser.add_argument("-m", "--check_doi", nargs = 1, default = "n", help = "Check if DOI is in PMC (y/n), default: no (n)")
args = parser.parse_args()
################################## Convert DOI to PMCID and PMID ########################################

# PMC API
API_KEY =  ".."  # NCBI API key nessessary for batch xml download
BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

# Function to fetch PMIDs & PMCIDs for a batch of DOIs
def fetch_ids_batch(doi_batch):
    """ This function fetches PMCIDS and PMIDS using batch
        resquest (doi_batch) to minimize the request number.
        This function recalls itself when the rate limit is
        exceeded, it retries after 10 seconds on the same
        batch that failed.
        The try-catch is to avoid stopping the script when an
        exeption arises.
    """
    params = {
        'format': 'json',
        'ids': ','.join(doi_batch),  # This is to send multiple DOIs in one request
        'api_key': API_KEY
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [(rec.get('doi', 'Not Found'), rec.get('pmid', 'Not Found'), rec.get('pmcid', 'Not Found'))
                    for rec in data.get('records', [])]
        elif response.status_code == 429:
            print("Rate limit exceeded! Retrying after 10 seconds...")
            time.sleep(10)
            return fetch_ids_batch(doi_batch)  # Retry the same batch
    except requests.RequestException as e:
        print(f"Request failed: {e}")

    # Return errors if request fails
    return [(doi, 'Error', 'Error') for doi in doi_batch]

def batch(dois):
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
	return result_df

# Extract results from API response & map to DOIs
def extract_results(api_response, doi_batch):
    found_dois = {record["doi"]: 1 for record in api_response.get("records", []) if "pmcid" in record}

    # Map input DOIs to found/not found status
    return {doi: found_dois.get(doi, 0) for doi in doi_batch}
    
# Function to check if a batch of DOIs exists in PMC
def check_doi_existence(doi_batch):
    params = {
        "format": "json",
        "ids": ",".join(doi_batch),  # Batch of DOIs
        "api_key": API_KEY
    }

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = requests.get(BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            return extract_results(response.json(), doi_batch)
        except requests.exceptions.Timeout:
            print(f"Timeout for batch {doi_batch}. Retrying ({attempt+1}/3)...")
            time.sleep(2 ** attempt)  # Exponential backoff (2s, 4s, 8s)
        except requests.exceptions.RequestException as e:
            print(f"Request failed for batch {doi_batch}: {e}")
            return {doi: -1 for doi in doi_batch}  # Mark all as failed

    return {doi: -1 for doi in doi_batch}  # Failure after all retries

# Process DOIs in parallel using 5 workers
def process_dois(dois, batch_size=150, workers=5):
    df_found = []
    df_not_found = []

    # Split DOIs into batches of `batch_size`
    doi_batches = [dois[i:i + batch_size] for i in range(0, len(dois), batch_size)]

    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(check_doi_existence, doi_batches))

    # Flatten results & store in DataFrames
    for batch_result in results:
        for doi, count in batch_result.items():
            if count > 0:
                df_found.append({"DOI": doi, "Found": count})
            else:
                df_not_found.append({"DOI": doi, "Found": count})

    # Convert to DataFrames & Save
    pd.DataFrame(df_found).to_csv('found_dois.csv', index=False)
    pd.DataFrame(df_not_found).to_csv('not_found_dois.csv', index=False)
    print(f"Found: {len(df_found)}, Not Found: {len(df_not_found)}")
    print(f"Saved found DOIs in {found_dois.csv} and not found in {not_found_dois.csv}") 
    return df_found, df_not_found
    
#   Load PMCIDs from TSV file
def load_pmc_ids(pmcid_df):
    print(pmcid_df)
    pmcid_df = pmcid_df.loc[pmcid_df['PMCID'] != 'Not Found']  # useless to look for the non existing pmcids
    return pmcid_df["PMCID"].astype(str).tolist()  # Ensure PMCIDs are strings

#   Download XMLs in batch using NCBI API
def download_pmc_xml_batch(pmc_ids, batch_size=200):
    """Fetch articles in batches from PMC using E-utilities."""
    epost_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    for i in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} with {len(batch)} IDs...")
        
        # Post IDs to Entrez
        epost_params = {"db": "pmc", "id": ",".join(batch),"api_key": API_KEY}
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
            "api_key": API_KEY
        }
        efetch_response = requests.get(efetch_url, params=efetch_params, timeout=30)
        efetch_response.raise_for_status()
        
        # Parse and save articles
        root = ET.fromstring(efetch_response.content)
        for article in root.xpath(".//article"):
            pmid = article.find(".//article-meta/article-id[@pub-id-type='pmc']")
            pmid_text = pmid.text if pmid is not None else "unknown"

            file_path = os.path.join(args.output, f"{pmid_text}.xml")
            with open(file_path, "wb") as f:
                f.write(ET.tostring(article, encoding="utf-8", pretty_print=True))
        print(f"Downloaded batch {i//batch_size + 1} with {len(batch)} xml files")

    retry_with_smaller_batches(pmc_batch)


#   Split PMCIDs into batches
def split_into_batches(pmc_ids, batch_size):
    return [pmc_ids[i:i+batch_size] for i in range(0, len(pmc_ids), batch_size)]

def retry_with_smaller_batches(pmc_batch, min_batch_size=5):
    """Recursively splits a batch into smaller sub-batches and retries downloads."""
    if len(pmc_batch) <= min_batch_size:
        print(f"Even with small batches, failed to download: {pmc_batch}")
        return  # Stop trying if the batch is too small

    mid = len(pmc_batch) // 2
    sub_batch1, sub_batch2 = pmc_batch[:mid], pmc_batch[mid:]

    print(f"Splitting batch into two smaller batches: {len(sub_batch1)} and {len(sub_batch2)}")
    download_pmc_xml_batch(sub_batch1)
    download_pmc_xml_batch(sub_batch2)

def batch(dois):
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
    return pd.DataFrame(results, columns=['DOI', 'PMID', 'PMCID'])

    # Load DOIs from CSV (non null and unique ones only)
if __name__ == "__main__":
    start = time.time()

    if args.convert:
        dois = pd.DataFrame(args.convert, columns=['DOI'])
    elif args.convert_file:
        df = pd.read_csv(args.convert_file[0], dtype=str)
        dois = df['DOI'].dropna().unique()
    
    pmcid_df = batch(dois)
    pmcid_df.to_csv(args.file, index=False)
    print(f"Saved PMCIDs/PMIDs in {args.output}")

    if args.check_doi and args.check_doi[0] == 'y':
        f, nf = process_dois(dois)

    if args.download and args.download[0] == 'y':
        # Ensure output directory exists
        if os.path.exists(args.output):  
            shutil.rmtree(args.output)
        os.makedirs(args.output, exist_ok=True)

        # Batch and parallel download
        pmc_ids = load_pmc_ids(pmcid_df)
        download_pmc_xml_batch(pmc_ids)

    end = time.time()
    print(f"Completed in {round(end - start, 2)} seconds.")
