import os
import time
import shutil
import requests
import argparse
import pandas as pd
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description ='DOIs to PMCIDs and xml download.')
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument("-f", "--convert_file", nargs = 1,  help ='Convert DOIs into PMCIDs/PMIDs from file')
group.add_argument("-c", "--convert", nargs = '+',  help ='Convert DOIs into PMCIDs/PMIDs from given list')
parser.add_argument("-t", "--output_file", nargs = 1, default = "dois_to_pmcids.tsv", help = "File name, default = dois_to_pmcids.tsv")
parser.add_argument("-o", "--output", nargs = 1, default = "pmc_xml", help = "Folder name, default = pmc_xml")
parser.add_argument("-d", "--download", nargs = 1, default = 'n', help ='Download the xml files of the DOIs having a PMCID (y/n), default: no (n)')
parser.add_argument("-m", "--check_doi", nargs = 1, default = "n", help = "Check if DOI is in PMC (y/n), default: no (n)")
args = parser.parse_args()
################################## Convert DOI to PMCID and PMID ########################################

# PMC API
API_KEY = '..'  # NCBI API key
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
    pmcid_df = pmcid_df.loc[df['PMCID'] != 'Not Found']  # useless to look for the non existing pmcids
    return df["PMCID"].astype(str).tolist()  # Ensure PMCIDs are strings

#   Download XMLs in batch using NCBI API
def download_pmc_xml_batch(pmc_batch):
    max_retries = 3
    pmc_list = ",".join(pmc_batch)  # Format PMCIDs as comma-separated list
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_list}&rettype=xml"

    for attempt in range(  max_retries):  # Retry logic
        try:
            response = requests.get(url, timeout=30)

            #  Handle 429 (Rate Limiting)
            if response.status_code == 429:
                print(f" Rate limited (429). Retrying in {2 ** attempt} seconds...")
                time.sleep(10 ** attempt)  # Use exponential backoff
                continue  # Retry

            #  Handle 200 (Success)
            if response.status_code == 200:
                for i, pmc_id in enumerate(pmc_batch):
                    file_path = os.path.join(args.output, f"{pmc_id}.xml")
                    with open(file_path, "wb") as f:
                        f.write(response.content)  # Save response XML
                #print(f" Batch download")
                return  # Exit function if successful

            #  Handle Other Errors 
            print(f" Failed batch {pmc_batch[:5]}... (Status {response.status_code})")
            break  # No retries for non-recoverable errors (e.g., 404)

        except requests.exceptions.Timeout:
            #print(f" Timeout for batch {pmc_batch}. Retrying ({attempt+1}/{ max_retries})...")
            time.sleep(2 ** attempt)  # Exponential backoff

        except requests.exceptions.RequestException as e:
            print(f" Request failed for batch: {e}")
            break  # Don't retry if it's a permanent failure

    print(f"Giving up on batch of size: {len(pmc_batch)}")
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

def parallel_download(pmc_ids):
    BATCH_SIZE = 150  
    MAX_WORKERS = 15
    pmc_batches = split_into_batches(pmc_ids, BATCH_SIZE)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(download_pmc_xml_batch, pmc_batches)

if __name__ == "__main__":
    start = time.time()

    if args.convert:
        dois = pd.DataFrame(args.convert, columns=['DOI'])
    elif args.convert_file:
        df = pd.read_csv('data/metadata.csv', dtype=str)
        dois = df['DOI'].dropna().unique()

    pmcid_df = batch(dois)
    pmcid_df.to_csv(args.output_file, index=False, sep='\t')
    print(f"Saved PMCIDs/PMIDs in {args.output_file}") 
    
    if args.check_doi[0] == 'y':
        f, nf = process_dois(dois)

    if args.download[0] == 'y':
        # Ensure XML directory exists
        if os.path.exists(args.output):  
            shutil.rmtree(args.output)
        os.makedirs(args.output, exist_ok=True)

        # Batch and parallel download
        pmc_ids = load_pmc_ids(pmcid_df)
        parallel_download(pmc_ids)

    end = time.time()
    print(f"Completed in {round(end-start, 2)} seconds.")

