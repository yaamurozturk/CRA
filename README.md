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
