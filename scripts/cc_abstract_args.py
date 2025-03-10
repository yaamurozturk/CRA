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
parser.add_argument("-o", "--output", nargs = 1, default = "cc_abstract.tsv", help = "Output file name, default = cc_abstract.tsv")
parser.add_argument("-c", "--context", nargs = 1, default = "basic", help = "Basic (basic) or with cos similarity (cos) citation context, default = basic")
parser.add_argument("-p", "--pmc", nargs = 1, default = "..", help = "NCBI key")
parser.add_argument("-e", "--el", nargs = 1, default = "", help = "Elsevier key for faster elsevier requests")
parser.add_argument("-m", "--mail", nargs = 1, default = "", help = "E-Mail adress for faster elsevier requests")
args = parser.parse_args()


url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
BASE_URL = "https://api.crossref.org/works/"

key = args.pmc[0]     # NCBI key
mail = args.mail[0]
API_KEY = args.el[0]  # Elsevier key

bs = 150 # batch size
par = 10 # workers
pmid = {} 
no_pmid = []
failed_dois = []
dois_without_abstract = []

# # **Extract all citations in xml file:  1. Remove all citations that have no DOI  2. Save a df with unique DOI**

def split_sentences(text):
    """Splits text into sentences."""
    return re.split(r'(?<=[.!?])\s+', text)

def extract_pmc_citations(xml_file):
    """Extract citations from a given PMC XML file using lxml for faster parsing."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        citing_doi = None
        for article_id in root.findall(".//article-id"):
            if article_id.get("pub-id-type") == "doi":
                citing_doi = article_id.text
                break

        citing_title = None
        title_elem = root.find(".//article-title")
        if title_elem is not None:
            citing_title = title_elem.text.strip() if title_elem.text else "No title"
               
        ref_dict = {}
        for ref in root.findall(".//ref"):
            ref_id = ref.get("id")
            title_elem = ref.find(".//article-title")
            doi_elem = ref.find(".//pub-id[@pub-id-type='doi']")
            ref_dict[ref_id] = {
                "title": title_elem.text.strip() if title_elem is not None and title_elem.text else "No title",
                "doi": doi_elem.text.strip() if doi_elem is not None and doi_elem.text else "No DOI",
            }

        data = []
        for paragraph in root.findall(".//p"):
            text = " ".join(paragraph.itertext()).strip()
            sentences = split_sentences(text)
            citation_matches = paragraph.findall(".//xref")

            for citation in citation_matches:
                citation_id = citation.get("rid")
                if citation_id in ref_dict:
                    sentence_index = next((i for i, s in enumerate(sentences) if citation.text and citation.text in s), None)
                    
                    if sentence_index is not None:
                        start_index = max(0, sentence_index - 2)
                        end_index = min(len(sentences), sentence_index + 1)
                        context_sentences = sentences[start_index:end_index]
                        cleaned_context = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\n.*?\n', '', " ".join(context_sentences)).strip()

                        data.append([
                            citing_doi,
                            citing_title,
                            citation_id,
                            cleaned_context,
                            ref_dict[citation_id]["title"],
                            ref_dict[citation_id]["doi"]
                        ])

        df = pd.DataFrame(data, columns=["Citing DOI", 'Citing title', "Citation ID", "Citation Context", "Cited Title", "DOI"])
        if (args.xml_file):
        	df = df.drop(columns = ["Citing title"])
        return df[df["DOI"] != "No DOI"]  # Drop rows with "No DOI"

    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
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
    print("Columns: ", df.columns) 
    return df

# # **Extract PMID:  1. Save DOIs with PMID in a dico  2. Save DOIs with no PMID in a List**

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
        return f"Request failed: {e}", f"Request failed: {e}"

    return f"Response status code: {response.status_code}", f"Response status code: {response.status_code}"


def process_batch(b):
    return fetch_ids_batch(b)  # Returns (pmid_dict, no_pmid_list)

# **Extract Abstract from PMC:   1. Using the pmids in the dict   2. Using epost for bulk queries**

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
    root = ET.fromstring(efetch_response.content)  # Again, using .content for bytes
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

USER_AGENT = f"YourAppName/1.0 (mailto:{mail})"
HEADERS = {"User-Agent": USER_AGENT}
MAX_WORKERS = 5  # Number of parallel requests

def get_abstract_elsevier_bulk(dois):
    abstracts = {}
    e = 0

    for doi in tqdm(dois, desc="Fetching Elsevier Abstracts", unit="DOI"):
        base_url = f"https://api.elsevier.com/content/article/doi/{doi}"
        headers = {
            "X-ELS-APIKey": API_KEY,
            "Accept": "application/json"
        }

        response = requests.get(base_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            try:
                abstract = data["full-text-retrieval-response"]["coredata"]["dc:description"]
                if abstract:
                	abstracts[doi] = abstract  
                else:
                	abstracts[doi] = "Not found"
                	e += 1
            except KeyError:
                abstracts[doi] = "Not found"
        else:
            abstracts[doi] = f"Not found"

    return  pd.DataFrame(list(abstracts.items()), columns=["DOI", "Abstract"]) , e

def merge(dfs):
	for d in dfs :
		if not d.empty:
			d = d.drop_duplicates(subset=['DOI'])

	# Merge the initial df with duplicates DOIs with the result abstracts from PMC, CR and EL 
	merged_df = pmc_citations_df.merge(abstracts_df, on="DOI", how="left") \
		                    .merge(metadata_df, on="DOI", how="left", suffixes=("", "_cr")) \
		                    .merge(elsevier_df, on="DOI", how="left", suffixes=("", "_el"))

	col = "Abstract" 
	# Retreiving the column with abstract 
	for c in [f"{col}", f"{col}_cr", f"{col}_el"]: 
		pmc_citations_df[c] = pmc_citations_df["DOI"].map(abstracts_df.set_index("DOI")[col])

	# Drop redundant columns after selection
	columns_to_drop = [f"{col}_cr"] +\
			  [f"{col}_el"]

	merged_df.drop(columns=columns_to_drop, inplace=True)
	print("merged length ", len(merged_df))
	merged_df = merged_df.dropna(subset = [col]) # remove citations where the abstract was not found
	return merged_df


def get_embedding(texts):
    """Generate embeddings for a batch of texts."""
    if isinstance(texts, str):  # Handle single string case
        texts = [texts]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move input to GPU

    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings.cpu().numpy()  # Move back to CPU for NumPy operations

def cosine_similarity_matrix(mat1, mat2):
    """Efficient cosine similarity computation between two matrices."""
    dot_product = np.dot(mat1, mat2.T)
    norm1 = np.linalg.norm(mat1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(mat2, axis=1, keepdims=True)
    return dot_product / (norm1 * norm2.T)

def extract_citation_context(df, similarity_threshold=0.7):
    """Extract dynamic citation context efficiently."""
    results = []
    abstract_cache = {}


    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Citations", unit="citation context", leave=False):
        citing_paragraph = row["Citation Context"]
        cited_abstract = row["Abstract"]

        if not isinstance(citing_paragraph, str) or not isinstance(cited_abstract, str):
            results.append(None)
            continue

        # Check cache for cited abstract embedding
        if cited_abstract in abstract_cache:
            cited_embedding = abstract_cache[cited_abstract]
        else:
            cited_embedding = get_embedding(cited_abstract)
            abstract_cache[cited_abstract] = cited_embedding  # Store in cache

        # Split paragraph into sentences
        sentences = citing_paragraph.split(". ")  # Simple sentence segmentation

        # Get embeddings in batch
        sentence_embeddings = get_embedding(sentences)

        # Compute similarity (vectorized)
        similarities = cosine_similarity_matrix(sentence_embeddings, cited_embedding)

        # Select sentences based on threshold
        dynamic_context = [sentences[i] for i in range(len(sentences)) if similarities[i] >= similarity_threshold]

        # Fallback: use full paragraph if no high-similarity sentences found
        results.append(" ".join(dynamic_context) if dynamic_context else citing_paragraph)

    df["Citation Context"] = results
    return df

start_time = time.time()

############################################## Extract citations from xml file ######################################
if (args.xml_file):
	pmc_citations_df = extract_pmc_citations(args.xml_file[0])
elif (args.xml_folder):
	pmc_citations_df = process_files(args.xml_folder)
	
unique_dois = pmc_citations_df.drop_duplicates(subset=['DOI'])
print(f'{len(pmc_citations_df)} citations found, {len(unique_dois)} are unique')

''' # Test subset
pmc_citations_df = pmc_citations_df.head(100)
unique_dois = unique_dois.head(100)'''

############################################ Extract pmids for the unique DOIs ##########################################
batches = [unique_dois['DOI'][i:i + bs] for i in range(0, len(unique_dois['DOI']), bs)]
with ThreadPoolExecutor(max_workers = par) as executor:
    results = list(executor.map(process_batch, batches))


for pmid_dict, no_pmid_list in results:
    pmid.update(pmid_dict) 
    no_pmid.extend(no_pmid_list) 
t = time.time() - start_time
print(f"Completed fetching pmids.")

############################################### Fetching abstracts from PMC, Crossref and lsevier #######################################
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
	    
    # Check Elsevier    
    if len(dois_without_abstract) > 0:
        abstracts , e = get_abstract_elsevier_bulk(dois_without_abstract)
        elsevier_df = pd.DataFrame(abstracts)
        print(f"Completed fetching {e} abstracts from Elsevier.")
        
dfs = [abstracts_df, metadata_df, elsevier_df]
final_df = merge(dfs)

######################################### Cos similarity citation context #####################################
if (args.context[0] == 'cos'):
    # These are here because they take some time to load, better for the CLA
    print('Initialisation torch, model and tokenizer...')  
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Load transformer model for embeddings (SPECTER)
    model_name = "allenai/specter"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move model to GPU to avoid finding multiple devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract citation contexts efficiently
    cc = extract_citation_context(final_df)
    cc.to_csv(args.output, sep='\t', index=False)
    print(f"Completed fetching {len(cc)} citation contexts")
	
######################################### Simple paragraph citation context #####################################
else:
	final_df.to_csv(args.output, sep = '\t', index=False)
print(f"Completed fetching {len(metadata_df) - c + e + p}  abstracts for {len(final_df)} in {round(time.time() - start_time, 2)} seconds.")
print(f'Results are saved in {args.output}')


