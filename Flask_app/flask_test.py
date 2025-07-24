import sys
import time
import csv
import requests
import sqlite3
from cc import *
from pmc import *
import pandas as pd
from flask import *
from fileinput import filename 
from distutils.log import debug 
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from  classification.classi import *

server = Flask(__name__)
server.secret_key = 'dev-secret-key-123'

BASE_DIR = "../" #os.path.dirname(os.path.abspath(__file__)) # current directory
db_path = os.path.join(BASE_DIR, "database")

df = pd.read_csv('../Retraction/retraction_watch.csv', dtype=str)
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], format='%m/%d/%Y %H:%M', errors='coerce')
df['Year'] = df['RetractionDate'].dt.year
df = df.dropna(subset=['Year'])
columns = ['Subject','Journal','Publisher','ArticleType','Year']

app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

app.layout = html.Div([
    html.H3(""),
    dcc.Dropdown(
        id='column-selector',
        options=[{'label': col, 'value': col} for col in columns],
        value='Category'
    ),
    dcc.Graph(id='histogram')
])

@app.callback(
    Output('histogram', 'figure'),
    Input('column-selector', 'value') 
)
def update_histogram(selected_column):
    if df[selected_column].dtype == 'object' or df[selected_column].nunique() < 20:
        top_values = df[selected_column].value_counts().nlargest(10).index
        filtered_df = df[df[selected_column].isin(top_values)]
        fig = px.histogram(
            filtered_df, 
            x=selected_column, 
            color_discrete_sequence=['#dc3545']
        )
    
    elif 'year' in selected_column.lower() and pd.api.types.is_numeric_dtype(df[selected_column]):
        filtered_df = df[df[selected_column] >= 2008].copy()  
        filtered_df.loc[:, selected_column] = filtered_df[selected_column].astype(int).astype(str)
        
        fig = px.histogram(
            filtered_df,
            x=selected_column,
            color_discrete_sequence=['#dc3545'],
            category_orders={selected_column: sorted(filtered_df[selected_column].unique())}
        )
    
    else:
        fig = px.histogram(
            df, 
            x=selected_column, 
            nbins=25,
            color_discrete_sequence=['#dc3545']
        )

    fig.update_layout(
        title={
            'text': f"Histogram of number of retracted papers per {selected_column}",
            'font': {'size': 18}
        },
        xaxis_title={
            'text': selected_column,
            'font': {'size': 14}
        },
        yaxis_title={
            'text': "Retracted Papers count",
            'font': {'size': 14}
        },
        bargap=0.2,
        hoverlabel={
            'font': {'color': 'black'}
        }
    )
    return fig

@server.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@server.route('/upload', methods=['POST'])
def upload_file():
    f = request.files.get('file')
    if f and f.filename:
        filename = f.filename
        result_df = extract_pmc_citations(f) 
        result = classif(result_df)
        result =  result[["Citing DOI",  "Citation ID", "CC paragraph", "Citation_context","Section", "IMRAD", "Publication Date","predicted_label","DOI", "Cited title"]]
        print(result)
        return render_template(
            "results.html",
            filename=filename,
            result_df=result, 
            column_names=result.columns.values,
            row_data=list(result.values.tolist()),
            zip=zip
        )
    return redirect('/')

@server.route('/citations', methods=['POST'])
def citations():
    doi = request.form.get('doi')
    pmid = request.form.get('pmid')
    finalc = pd.DataFrame()
    pmc_citations = pd.DataFrame()

    """if (pmid) and (doi):
        pmcids = convert([pmid])
        print(pmcids)
        count , path = fetch_pubmed_articles(pmcids)
        print(f'{count} xml files were downloaded from PMC')
        pmc_citations_df = process_files(path)
        #Citing papers
    el"""
    if doi:
        if pmid:
            download_pmc_xml(pmid)
            file = 'citing_'+pmid#+ids['PMID'][0]
            print(file)
            pmids = extract_pmids(file)
            print(pmids)        
            pmcids = convert(pmids)
            print(pmcids)
            count_citing , path = fetch_pubmed_articles(pmcids)
            print(f'{count_citing} xml files were downloaded from PMC')
            citing_df = process_files(path)
            print(citing_df)
            #shutil.rmtree(path)
            final = citing_df[(citing_df['DOI'] == doi)]#|(pmc_citations_df['Cited title'].apply(lambda x: match_min_phrase(x, title) if pd.notna(x) else False))]
            final =  final[["Citing DOI",  "Citation ID", "CC paragraph", "Citation_context","Section", "IMRAD", "Publication Date","DOI", "Cited title"]]
            finalc = final #classif(final)
            print(finalc)  
        ids = doi_to_pmcid([doi])
        print(ids)
        pmcid = ids[ids.PMCID != 'Not Found']
        if not pmcid.empty:
            p = pmcid.PMCID#.head(5)
            count_cited , path = fetch_pubmed_articles(p)
            print(f'{count_cited} xml files were downloaded from PMC')
            print(path)
            pmc_citations_df = process_files(path)
            print(pmc_citations_df)
            #shutil.rmtree(path)

        else:
            crawlForPaper(doi)
            #xml = path+'/doi.xml'
            pmc_citations_df = extract_elsevier_citations('doi.xml')

    #result_cited =  result_cited[["Citing DOI",  "Citation ID", "CC paragraph", "Citation_context","Section", "IMRAD", "Publication Date","predicted_label","DOI", "Cited title"]]

    #classif(pmc_citations_df)
    if not final.empty:
        finalc = final
        result_citing = classif(finalc)
    else:
        result_citing = pd.DataFrame()

    if not pmc_citations_df.empty:
        pmc_citations = pmc_citations_df
        result_cited = classif(pmc_citations)
    else:
        result_cited = pd.DataFrame()
        

    # final_df = dois_cited_eval(doi_list, "try_pipeline")# pd.read_csv("../../Citing/abstracts_like_vickers.tsv", sep = '\t', dtype = str) #
    

    
    #print(result_citing.columns)
    print(pmc_citations)
    #result_citing =  result_citing[["Citing DOI",  "Citation ID", "CC paragraph", "Citation_context","Section", "IMRAD", "Publication Date","predicted_label","DOI", "Cited title"]]
    
    print(result_citing.columns)
    #result_cited = result_cited[["Citing DOI",  "Citation ID", "CC paragraph", "Citation_context","Section", "IMRAD", "Publication Date","predicted_label","DOI", "Cited title"]]

    print(len(result_cited),len(result_citing))

    return render_template(
        "cc.html", 
        results=result_cited.values.tolist(), 
        resu=result_citing.values.tolist(), 
        doi=doi,
        cited=len(pmc_citations),
        citing=len(finalc),
        column_names=result_cited.columns.tolist(),
        citing_count = len(finalc['Citing DOI'].dropna().unique()) if 'Citing DOI' in finalc.columns and not finalc['Citing DOI'].dropna().empty else 0,
        cited_count = len(pmc_citations['DOI'].dropna().unique()) if 'DOI' in pmc_citations.columns and not pmc_citations['DOI'].dropna().empty else 0,
        resu_column_names=result_citing.columns.tolist(),
    )

@server.route('/retraction', methods=['POST'])
def retraction():
    doi = request.form.get('doi')
    pmid = request.form.get('pmid')
    merged_citing = pd.DataFrame()
    merged_df = pd.DataFrame()
    pmc_citations_df = pd.DataFrame()
    
    if doi:
        print(doi)
                # Check retraction status
        retraction_df = pd.read_csv('retraction_watch.csv', dtype=str)
        retracted = doi in retraction_df['OriginalPaperDOI'].values
        reason = retraction_df.loc[retraction_df['OriginalPaperDOI'] == doi, 'Reason'].values[0] if retracted else ""
        reason_list = [item.strip() for item in reason.replace('+', '').split(';') if item.strip()]

        oa = check_open_access(doi)

        # get citing papers if the doi is retracted
        
            
        if pmid:
            download_pmc_xml(pmid)
            file = 'citing_'+pmid#+ids['PMID'][0]
            print(file)
            pmids = extract_pmids(file)
            #print(pmids)        
            pmcids = convert(pmids)
            #print(pmcids)
            count_citing , path = fetch_pubmed_articles(pmcids)
            print(f'{count_citing} xml files were downloaded from PMC')
            citing_df = process_files(path)
            #print(citing_df.columns.tolist())
            if retracted:
                merged_citing = citing_df.merge(retraction_df[['OriginalPaperDOI','Reason','RetractionDate']], left_on="Citing DOI", right_on='OriginalPaperDOI', how='inner').drop(columns=['OriginalPaperDOI'])   
                merged_citing = merged_citing if not merged_citing.empty else pd.DataFrame()
                
                #print(merged_citing)

        print('merged_citing,   ',merged_citing)

        ids = doi_to_pmcid([doi])
        print(ids)
        pmcid = ids[ids.PMCID != 'Not Found']
        if not pmcid.empty:
            p = pmcid.PMCID#.head(5)
            count , path = fetch_pubmed_articles(p)
            print(f'{count} xml files were downloaded from PMC')
            #print(path)
            pmc_citations_df = process_files(path)
            #print(pmc_citations_df)
            shutil.rmtree(path)

        if retracted:
            merged_df = pmc_citations_df.merge(retraction_df[['OriginalPaperDOI','Reason','RetractionDate']], left_on='DOI', right_on='OriginalPaperDOI', how='inner').drop(columns=['OriginalPaperDOI'])
        else:
            merged_df = merged_df if not merged_df.empty else pd.DataFrame()
        print(merged_df)
        
        if merged_df.empty:
            data = []
            columns = []
        else:
            data = merged_df.values.tolist()
            columns = merged_df.columns.tolist()
        print(merged_citing)    
        return render_template(
            "db.html",
            resu=data,  
            doi=doi,
            resu_column_names= columns,
            retracted=retracted,
            reasons=reason_list,
            merged_citing=merged_citing.values.tolist(),
            merged_citing_cols=merged_citing.columns.tolist(),
            citing_count= len(merged_citing['DOI'].unique()) if not merged_citing.empty else '0',
            cited_count= len(merged_df['DOI'].unique()) if not merged_df.empty else '0' ,
            citing=len(merged_citing),
            cited=len(data),
            oa=oa
        )



    
if __name__ == '__main__': 
    app.run( debug=True,host='0.0.0.0', port=5000,) 
