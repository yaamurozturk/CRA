# Ongoing work: Citations to retracted articles (CRA) data extractor and analyzer tool to help in post publication peer review. 

Contributors: Tiziri Terkmani, Yagmur Ozturk, Qinyue Liu 

# Static website link: TBA, undergoing server updates
<img width="1510" height="770" alt="resim" src="https://github.com/user-attachments/assets/e00c6cb4-7cfa-48b1-ab9f-5d74d4129926" />


# Citation Context Extraction:  

This repository containes scripts for **citation context extraction**, with support for complex and stacked citation handling in pmc xml files.

-  **Extract citation contexts** from PMC XML files.
-  **Handle stacked citations** such as `[1–4]`, `[3, 5–7]`, or mixed `<xref>` tag structures.
-  **Convert identifiers** between DOI, PMID, and PMCID.
-  **Download full-text PMC XML files** using PMCID/PMID.
-  **Fetch abstracts** from:
    - **Crossref**
    - **PubMed Central (PMC)**
    - **Elsevier API**
- **Extract citation contexts** from citing articles referencing a specific paper (via DOI, PMID, or PMCID).
- **Retrieve citation context metadata**, including section titles and IMRAD structure (Introduction, Methods, Results, Discussion).


## Get results table with ["DOI", "Cited title", "Citation ID", "CC paragraph", "Citation_context", "Citing DOI", "PMCID"]:

  Use Stacked_citations/Stacked_citations_simple_sent.py

### Exapmle usage command line:

- python3 Stacked_citations_simple_sent.py -x path_to_pmc_xml_file.xml      **for one file**
- python3 Stacked_citations_simple_sent.py -f path_to_pmc_xml_directory     **for folder**


## Flask_app folder:
  * Source code for the Flask APP.
  * For now it only works with one DOI input at a time.
  * For a smooth, run the libraries in **requirements.txt** file need to be installed using **pip install -r requirements.txt**.
  * **Retraction_watch.csv file should be _downloaded_ and put inside the Flask_app directory**: Dowload link: https://gitlab.com/crossref/retraction-watch-data

### Usage:
    python3 flask_test.py 
    The app should be running on http://0.0.0.0:5000/
    
