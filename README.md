# Citations to retracted articles (CRA) Analysis Tool for Post Publication Peer Review

Disclaimer: This is an ongoing work which will be updated regularly, thanks for your patience! 

## Contributors: Yagmur Ozturk, Tiziri Terkmani, Qinyue Liu , Cyril Labbé

Citations to retracted publications need to be analyzed to ensure the reliability of citing literature since they are unreliable sources. However, identfying these citations and finding in which contexts they are cited is a time consuming task. Here, we offer a pipeline that helps with this task! 

We combine metadata that we find necessary to analyse CRA such as retraction reasons*, retraction & publication date, citation contexts (sentence based and larger windows), highlighting of the retracted reference inside the context (especially useful when there are multiple references). All of this information can be visualized in a table to offer easy access & PPPR after running the pipeline (or using the link above). All you need to provide is the PMID and the DOI of the publication that you're interested in analysing. 
The pipeline currently only supports the parsing of Pubmed Central Open Access (PMC-OA) articles with full-text XMLs, but we are working (hard) to support other formats!

# Static website link: TBA, undergoing server updates
<img width="1671" height="748" alt="resim" src="https://github.com/user-attachments/assets/2588de52-0af6-4eef-a9a4-640e2591aa41" />




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

## Sources
- Problematic Paper Screener' Feet of Clay Detector: https://dbrech.irit.fr/pls/apex/f?p=9999:31::::::
- COSIG's guide on citations to retracted publications and how to do PPPR on them: https://osf.io/9q3as
- PMC OA: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- *We obtain Retraction Reasons and other retraction metadata obtained from the Retraction Watch Database. Thank you Retraction Watch! https://retractiondatabase.org/RetractionSearch.aspx?

## Acknowledgments 
We acknowledge the NanoBubbles project that has received Synergy grant funding from the European Research Council (ERC), within the European Union’s Horizon 2020 program, grant agreement no. 951393. 


<img width="131" height="132" alt="resim" src="https://github.com/user-attachments/assets/9200c047-183d-457d-a4b8-0260005f6d5b" />
<img width="265" height="188" alt="resim" src="https://github.com/user-attachments/assets/ee5c8f12-3c5c-4bbe-b957-d454dd304a84" />

