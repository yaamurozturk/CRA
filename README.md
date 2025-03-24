# Citation Context Extraction Toolkit

This repository provides tools for advanced **citation context extraction**, with support for complex and stacked citation handling.

## Features

-  **Extract citation contexts** from PMC XML files.
-  **Handle stacked citations** such as `[1–4]`, `[3, 5–7]`, or mixed `<xref>` tag structures.
-  **Convert identifiers** between DOI, PMID, and PMCID.
-  **Download full-text PMC XML files** using PMCID or PMID.
-  **Fetch abstracts** from:
    - **Crossref**
    - **PubMed Central (PMC)**
    - **Elsevier API**
- **Extract citation contexts** from citing articles referencing a specific paper (via DOI, PMID, or PMCID).
- **Retrieve citation context metadata**, including section titles and IMRAD structure (Introduction, Methods, Results, Discussion).

## Usage Highlights

- **Batch process folders** of PMC XML files for citation context extraction.
- Handle edge cases like:
  - Implicit citation markers (e.g., `[12–14]` spanning multiple `<xref>` tags)
  - Implicit citation markers (e.g., `[12–14]` in a simple `<xref>` tag)
  - Implicit citations: `[5, 12–14 ; 6]`

## Example Citation context extraction

1. **Input**: PMC XML full-text article or batch of XML files.
2. **Process**: 
   - Identify inline citations via `<xref>` tags.
   - Expand stacked citations automatically.
   - Extract surrounding sentence, paragraph, and section context.
3. **Output**: Structured data frame with citation metadata.



# PMC citations 

## PMC_scripts.py with command line arguments:
- This script converts DOIs into PMCIDs/PMIDs, checks if a given DOI is in PMC and downloads the xml files corresponding to the given DOIs. 
1. **Convert DOIs** (from a csv file or from a list given in the arguments) outputs it into a tsv file : 
- **Example usage:**
- python3 PMC_scripts.py -f path_to/csv_file_containing DOIs
- python3 PMC_scripts.py -c doi1 doi2 ... -t for the tsv output file name (default = dois_to_pmcids.tsv)

2. **Download the xml files corresponding to DOIs that have a PMCID** 
- **Example usage:**
- python3 PMC_scripts.py -f path_to/doi_file -d y (default value is n) -o xml dowload folder name (default = pmc_xml)

3. **Check if DOI is in PMC:**
- **Example usage:**
- python3 PMC_scripts.py -f path_to/doi_file -m y
    
##  PMC citations and abstract extraction cc_abstract_args.py: 
- This script extracts all the citations inside an xml file, their DOI, and the cc (the whole paragraph). Then extracts the corresponding abstract from PMC -> CrossRef -> Elsevier. There is an optimized citation context option for a (slightly) more accurate CC (working on this with dynamic CC).

1. **Input:** xml file or folder containing multiple xml files:
- **Example usage:**
- python3 cc_abstract_args.py -x path_to/xml_file 
- python3 cc_abstract_args.py -f path_to/xml_folder  

2. **Get optimized cc (with cos similarity):** Withou this option the cc returned is basic (the paragraph containing the citation).
- **Example usage:**
- python3 cc_abstract_args.py -xml path_to/file_name -c cos

3. **API Key:**
- Add NCBI and Elsevier key for fatser requests (and an e-mail adress for elsevier):
- **Example usage:**
- python3 cc_abstract_args.py -x path_to/xml_file -p pmc_key -e elsevier_key -m e-mail_@
