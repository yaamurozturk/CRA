import os
import re
import time
import json
import random
import shutil
import requests
import pandas as pd
from tqdm import tqdm
import lxml.etree as ET
from itertools import islice
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
missing_pmids = []
pmids = []
key = "fd895b77ece1cd582d9d2a40cc6d23f88008"
apiKey = '4d679c4998d5031c0b6a86e72f1b8c87'
instToken = "ed2c69e1fa68fc3cba8fda583c8e4b25"
path = os.getcwd()+"elsevier_xml"

current_dir = os.getcwd()

def crawlForPaper(doi,outputRep):
    print("Try to crawl: "+doi)
    #Publisher = re.sub('/.*','',doi)
    folder = path#outputRep+"/elsevier_xml"
    #print("Publisher:"+Publisher)
    if not os.path.isdir(folder):
        print("Creating:"+folder)
        os.mkdir(folder)
    else:
        print ("Publisher subfolder exists")

    if "10.1016/" in doi:
        dstName = folder+"doi.xml"
        if not os.path.isfile(dstName):
            resp = requests.get("https://api.elsevier.com/content/article/doi/"+doi, params={"apiKey": apiKey, "instToken": "ed2c69e1fa68fc3cba8fda583c8e4b25", "httpAccept": "text/xml", "view": "FULL"})
            #resp.raise_for_status
            with open(dstName, 'w') as f:
                f.write(resp.text)
                print("Wrote: '"+dstName+"'")
        else:
            print("XML is already here: "+dstName)
    else:
        print("Not an Elsevier paper: "+doi)


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
    batch_size = 20
    results = {}
    j = 0

    def request_dois(ids):
        params = {'format': 'json', 'ids': ','.join(ids), 'api_key': key}
        try:
            response = requests.get(BASE_URL, params=params, timeout=20)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limit exceeded on batch!")
            else:
                print(f"Failed request with status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        return None

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        print(f'Requesting batch {j}')
        data = request_dois(batch)

        if data:
            invalid_pmids = []
            for rec in data.get('records', []):
                #print(rec)
                if 'errmsg' in rec and 'invalid article id' in rec['errmsg'].lower():
                    invalid_pmids.append(f"PMC{rec.get('pmid', '')}")
                else:
                    pmid = rec.get('pmid')
                    doi = rec.get('doi', pmid)
                    if pmid:
                        results[pmid] = doi

            if invalid_pmids:
                print(f"Retrying {len(invalid_pmids)} invalid PMIDs as PMC IDs")
                retry_data = request_dois(invalid_pmids)
                if retry_data:
                    for rec in retry_data.get('records', []):
                        #print(rec)
                        pmcid = rec.get('pmcid', '').replace('PMC', '')
                        doi = rec.get('doi', pmcid)
                        print(pmcid, doi)
                        if pmcid:
                            results[pmcid] = doi
        j += 1

    return results

def split_sentences(text):
    protected_abbr = ["et al.","e.g.","i.e.","Fig.","Ref.","No.","cf.","ca.","vs.","Dr.","Prof.","Mr.","Mrs.","Ms.",
                      "Inc.","Ltd.","Co.","Eq.","Vol.","pp.","Ch.","eds.","ed.","al.","Jr.","Sr.","St.","mg.", "mL.", "cm.", "mm.", "kg.", "km."]
    for abbr in protected_abbr:
        text = text.replace(abbr, abbr.replace(".", "<DOT>"))
    text = re.sub(r'\[(?:[^\[\]]*?[.;][^\[\]]*?)\]', lambda m: m.group(0).replace('.', '<DOT>').replace(';', '<SEMI>'), text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.replace('<DOT>', '.').replace('<SEMI>', ';') for s in sentences]
    return sentences


def parse_components(ref_id):
    match = re.match(r"([a-zA-Z]*)(\d+)([a-z]?)", ref_id)
    return match.groups() if match else (None, None, None)


def expand_citation_range(start_id, end_id, ref_list):
    start_prefix, start_num, start_suffix = parse_components(start_id)
    end_prefix, end_num, end_suffix = parse_components(end_id)
    if start_prefix != end_prefix or not start_num or not end_num:
        return [start_id, end_id]

    try:
        start_num = int(start_num)
        end_num = int(end_num)
    except ValueError:
        return [start_id, end_id]

    if start_num > end_num or (end_num - start_num) > 25:
        return [start_id, end_id]

    expanded = []
    for ref in ref_list:
        prefix, num, suffix = parse_components(ref)
        if prefix == start_prefix and num and start_num <= int(num) <= end_num:
            expanded.append(ref)
    return sorted(expanded, key=lambda x: int(parse_components(x)[1]))


def seen_sent(seen, sentences, citation_text, rid, data, ref_dict, ref_id, full_text, citing_doi, pmcid):  
    for i,s in enumerate(sentences):
        if (citation_text and citation_text in s) or rid in s:
            sent = ' '.join(sentences[max(i-1,0): min(len(sentences),i +2)])
            #sent = re.sub(r'[^a-zA-Z0-9., %]', '', sent, flags=re.ASCII)
            matching_sentences = [sent]
            
    if not matching_sentences:
        matching_sentences = [full_text]
    
    for sentence in matching_sentences:
        if (rid, sentence) in seen:
            continue 
        seen.add((rid, sentence))
        data.append(create_citation_row(ref_dict, rid, full_text, sentence, citing_doi, pmcid))

def extract_elsevier_citations(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        ns = {
            'ce': 'http://www.elsevier.com/xml/common',
            'dtd': 'http://www.elsevier.com/xml/common/dtd',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'mt': 'http://www.elsevier.com/xml/common/struct-bib/dtd'
        }


        citing_doi = root.findtext('.//dc:identifier', default='No DOI', namespaces=ns)
        print(citing_doi)

        ref_dict, ref_list = {}, []
        
        #for elem in root.iter():
         #   print(elem.tag)
        #refs = root.findall('.//{http://www.elsevier.com/xml/xocs/dtd}bib-reference') 
        
        for ref in root.findall('.//dtd:bib-reference',namespaces=ns):
            #print(ref)
            ref_id = ref.get('id')
            #print(ref_id)
            if not ref_id:
                continue
            ref_list.append(ref_id)

            title_elem = ref.find('.//mt:maintitle', namespaces=ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title"

            doi_elem = ref.find('.//dtd:doi', namespaces=ns)
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

            ref_dict[ref_id] = {
                "title": title,
                "doi": doi if doi else "No DOI",
            }
        #print(ref_list)
        data = []
        seen = set()
        for para in root.findall('.//dtd:para', namespaces=ns):
            full_text = " ".join(para.itertext()).strip()
            if not full_text:
                continue

            sentences = split_sentences(full_text)
            xrefs = para.findall('.//dtd:cross-ref', namespaces=ns)

            i = 0
            while i < len(xrefs):
                rid = xrefs[i].get("refid")
                citation_text = xrefs[i].text

                seen_sent(seen, sentences, citation_text, rid, data, ref_dict, rid, full_text, citing_doi,'Not a Pubmed paper')
                i += 1

        return pd.DataFrame(data, columns=["DOI", "Cited title", "Citation ID", "CC paragraph", "Citation_context", "Citing DOI","PMCID"])

    except Exception as e:
        print(f"Error processing file {xml_file}: {str(e)}")
        return pd.DataFrame()

def extract_pmc_citations(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        article = next(root.iter("article"), None)
        if article is None:
            return pd.DataFrame()

        citing_doi = pmcid = None
        for article_id in article.findall(".//article-id"):
            if article_id.get("pub-id-type") == "doi":
                citing_doi = article_id.text
            if article_id.get("pub-id-type") == "pmc":
                pmcid = article_id.text

        ref_dict, ref_list = {}, []
        for ref in article.findall(".//ref"):
            ref_id = ref.get("id")
            if not ref_id:
                continue
            ref_list.append(ref_id)
            title_elem = ref.find(".//article-title")
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title"
            doi_elem = ref.find(".//pub-id[@pub-id-type='doi']")
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None
            pmid_elem = ref.find(".//pub-id[@pub-id-type='pmid']")
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None
            ref_dict[ref_id] = {
                "title": title,
                "doi": doi if doi else pmid if pmid else "No DOI",
                "pmid": pmid
            }

        data = []
        seen = set()
        for paragraph in article.findall(".//p"):
            full_text = " ".join(paragraph.itertext()).strip()
            if not full_text:
                continue
            sentences = split_sentences(full_text)
            xrefs = [elem for elem in paragraph.iter() if elem.tag == "xref" and elem.get("ref-type") == "bibr"]
            i = 0
            while i < len(xrefs):
                rid = xrefs[i].get("rid")
                citation_text = xrefs[i].text
                #print(rid, citation_text)
                # Check for stacked range pattern: xref - xref
                if i+1 < len(xrefs):
                    mid_text = (xrefs[i].tail or '').strip()
                    if mid_text in ['-', '–']:
                        end_rid = xrefs[i+1].get("rid")
                        expanded = expand_citation_range(rid, end_rid, ref_list)
                        for ref_id in expanded:
                            seen_sent(seen,sentences, citation_text, rid, data, ref_dict, ref_id, full_text, citing_doi, pmcid)
                            #data.append(create_citation_row(ref_dict, ref_id, full_text, sentence, citing_doi, pmcid))
                        i += 2
                        continue
                # Check for stacked individual citations: <xref/><xref/><xref/>
                if any(sep in (xrefs[i].tail or '') for sep in [',', ';']):
                    seen_sent(seen,sentences, citation_text, rid, data, ref_dict, ref_id, full_text,  citing_doi, pmcid)
                    #if rid == 'B45':
                     #   print(sentence)
                    
                    i += 1
                    continue

                # Single xref
                seen_sent(seen,sentences, citation_text, rid, data, ref_dict, ref_id, full_text, citing_doi, pmcid)

                #data.append(create_citation_row(ref_dict, rid, full_text,  citing_doi, pmcid))
                i += 1

        return pd.DataFrame(data, columns=["DOI", "Cited title", "Citation ID", "CC paragraph", "Citation_context", "Citing DOI", "PMCID"])

    except Exception as e:
        print(f"Error processing file {xml_file}: {str(e)}")
        return pd.DataFrame()



def create_citation_row(ref_dict, ref_id, text, sentence, citing_doi, pmcid):
    ref_info = ref_dict.get(ref_id, {"title": "No title", "doi": "No DOI"})
    return [
        ref_info["doi"],
        ref_info["title"],
        ref_id,
        text,
        sentence,
        citing_doi,
        pmcid
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
                df = future.result()
                if not df.empty:
                    filename = os.path.basename(file)
                    num_citations = len(df)
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        total_citations = len(combined_df)
        print(f"\nTotal: Extracted {total_citations} citations from {len(all_dfs)} files in {time.time() - start:.2f} seconds")
        return combined_df#, all_missing_pmids
    else:
        print("No citation data extracted from any file.")
        return pd.DataFrame()#, []
        
"""############ Fetch PMIDS ##################"""

def fetch_ids_batch(doi_batch, attempt=1, BACKOFF_FACTOR=2, MAX_RETRIES=3):
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
            #logging.warning(f"Rate limit exceeded! Retrying after {wait_time} seconds (attempt {attempt})...")
            time.sleep(wait_time)
            if attempt < MAX_RETRIES:
                return fetch_ids_batch(doi_batch, attempt + 1)
        elif response.status_code in {500, 502, 503, 504}:
            wait_time = BACKOFF_FACTOR * attempt
            #logging.warning(f"Server error {response.status_code}! Retrying after {wait_time} seconds (attempt {attempt})...")
            time.sleep(wait_time)
            if attempt < MAX_RETRIES:
                return fetch_ids_batch(doi_batch, attempt + 1)
        else:
            print(f"Request failed with status {response.status_code}: {response.text}")

    except requests.RequestException as e:
        #logging.error(f"Request exception: {e}")
        if attempt < MAX_RETRIES:
            wait_time = BACKOFF_FACTOR * attempt
            #logging.warning(f"Retrying after {wait_time} seconds (attempt {attempt})...")
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
    #result_df.to_csv('doi_to_pmcids.tsv', index=False, sep='\t')

    # Record end time
    end = time.time()

    #print(f"Completed in {round(end-start, 2)} seconds.")
    return result_df

def convert(pmids):
    batch_size = 150  # Number of DOIs per request
    nb_workers = 5  # Number of parallel requests
    results = []

    # Split DOIs into batches of 10
    batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]

    # Execute requests in parallel using ThreadPoolExecutor with 5 workers
    with ThreadPoolExecutor(max_workers = nb_workers) as executor:
        results_list = list(executor.map(fetch_ids_batch, batches)) # This returns a list of lists (batches) of tuples

    # Flatten results (since each batch returns a list)
    results = [item for sublist in results_list for item in sublist] # This is a list of tuples after flattening the batches lists
    result_df = pd.DataFrame(results, columns=['DOI', 'PMID', 'PMCID'])
    return result_df.PMCID

def normalize_dashes(text):
    """Convert all dash-like characters to standard hyphens."""
    return re.sub(r'[‐‒–—―−]', '-', str(text).lower())

def match_min_phrase(cited_title, target_title):
    """Check if cited_title contains the first critical part of target_title."""
    if pd.isnull(cited_title) or pd.isnull(target_title):
        return False
    min_phrase = "Protease inhibitors from marine actinobacteria as a potential"
    # Normalize both strings
    normalized_cited = normalize_dashes(cited_title)
    normalized_phrase = normalize_dashes(min_phrase)
    
    return normalized_phrase in normalized_cited


"""############### Download PubMed XML files #########""" 

def download_pmc_xml(pmid): # To get the xml for the papers citing the given DOI
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pubmed_citedin&id={pmid}&rettype=xml"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
       
        file_path = f"{pmid}.xml"
        with open(f"citing_{pmid}", "wb") as f:
            f.write(response.content)  # Save response XML
        print(f" File downloaded")
        return  # Exit function if successful

    #  Handle Other Errors 
    print(f" Failed ... (Status {response.status_code})")


def extract_pmids(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    #print(root.findall(".//Link"))
    for link in root.findall(".//Id"):
        text = " ".join(link.itertext()).strip()
        pmids.append(text)
    return pmids

def fetch_pubmed_articles(pmc_ids, batch_size=100):
    """Fetch articles in batches from PMC using E-utilities."""
    count = 0  # Successfully downloaded xml files
    if os.path.exists(path):  
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    epost_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    for i in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} with {len(batch)} IDs...")
        
        try:
            # Post IDs to Entrez
            epost_params = {"db": "pmc", "id": ",".join(batch), "api_key": key}
            epost_response = requests.post(epost_url, data=epost_params, timeout=10)
            epost_response.raise_for_status()
            
            # Parse WebEnv and QueryKey with error handling
            root = ET.fromstring(epost_response.content)
            webenv_element = root.find(".//WebEnv")
            query_key_element = root.find(".//QueryKey")
            
            if webenv_element is None or query_key_element is None:
                print(f"Failed to get WebEnv/QueryKey for batch {i//batch_size + 1}. Response: {epost_response.text}")
                continue
                
            webenv = webenv_element.text
            query_key = query_key_element.text
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
            articles_root = ET.fromstring(efetch_response.content)
            for article in articles_root.xpath(".//article"):
                pmid = article.find(".//article-meta/article-id[@pub-id-type='pmc']")
                pmid_text = pmid.text if pmid is not None else f"unknown_{count}"
                
                file_path = os.path.join(path, f"{pmid_text}.xml")
                with open(file_path, "wb") as f:
                    f.write(ET.tostring(article, encoding="utf-8", pretty_print=True))
                count += 1
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed for batch {i//batch_size + 1}: {str(e)}")
            continue
        except ET.ParseError as e:
            print(f"XML parsing failed for batch {i//batch_size + 1}: {str(e)}")
            continue
        except Exception as e:
            print(f"Unexpected error processing batch {i//batch_size + 1}: {str(e)}")
            continue
            
    return count

