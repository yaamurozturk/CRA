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

server = Flask(__name__)
server.secret_key = 'dev-secret-key-123'

BASE_DIR = "../" #os.path.dirname(os.path.abspath(__file__)) # current directory
db_path = os.path.join(BASE_DIR, "database")

df = pd.read_csv('../Retraction/retraction_watch.csv', dtype=str)
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], format='%m/%d/%Y %H:%M', errors='coerce')
df['Year'] = df['RetractionDate'].dt.year
df = df.dropna(subset=['Year'])
columns = ['Journal','Publisher','ArticleType','Year']

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

def get_db():
    """Ensure database exists and get connection"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found.")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@server.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@server.route('/upload', methods=['POST'])
def upload_file():
    f = request.files.get('file')
    if f and f.filename:
        filename = f.filename
        result_df = extract_pmc_citations(f) 
        return render_template(
            "results.html",
            filename=filename,
            result_df=result_df, 
            column_names=result_df.columns.values,
            row_data=list(result_df.values.tolist()),
            zip=zip
        )
    return redirect('/')

@server.route('/citations', methods=['POST'])
def citations():
    doi = request.form.get('doi')
    if doi:
        ids = doi_to_pmcid([doi])
        print(ids)
        pmcid = ids[ids.PMCID != 'Not Found']
        if not pmcid.empty:
            p = pmcid.PMCID#.head(5)
            count = fetch_pubmed_articles(p)
            print(f'{count} xml files were downloaded from PMC')
            pmc_citations_df = process_files(path)
            print(pmc_citations_df)
        else:
            crawlForPaper(doi,current_dir)
            pmc_citations_df = extract_elsevier_citations('doi.xml')
        shutil.rmtree(path)
        return render_template(
            "results.html",
            filename=doi,
            result_df=pmc_citations_df, 
            column_names=pmc_citations_df.columns.values,
            row_data=list(pmc_citations_df.values.tolist()),
            zip=zip
        )
    return redirect('/')


@server.route('/citing', methods=['POST'])
def citing():
    doi = request.form.get('doi')
    title = request.form.get('title')
    if doi:
        ids = doi_to_pmcid([doi])
        print(ids)
        download_pmc_xml(ids['PMID'][0])
        file = 'citing_'+ids['PMID'][0]
        print(file)
        
        pmids = extract_pmids(file)
        print(pmids)        
        pmcids = convert(pmids)
        print(pmcids)
        count = fetch_pubmed_articles(pmcids)
        print(f'{count} xml files were downloaded from PMC')
        pmc_citations_df = process_files(path)
        print(pmc_citations_df)
        shutil.rmtree(path)
        final = pmc_citations_df[(pmc_citations_df['DOI'] == doi)|(pmc_citations_df['Cited title'].apply(lambda x: match_min_phrase(x, title) if pd.notna(x) else False))]
        return render_template(
            "results_citing.html",
            doi=doi,
            result_df=final, 
            column_names=final.columns.values,
            row_data=list(final.values.tolist()),
            zip=zip
        )
    return redirect('/')


@server.route('/process_doi', methods=['POST'])
def process_doi():
    doi = request.form.get('doi')
    if doi:
        pmc_data = fetch_ids_batch(doi)
        formatted_data = []
        for item in pmc_data:
            formatted_data.append({
                'DOI': item[0],
                'PMID': item[1],
                'PMCID': item[2]
            })
        
        return render_template('doi.html', 
                            pmc_data=formatted_data,
                            show_pmc_table=bool(pmc_data))
    return redirect('/')

@server.route('/database', methods=['POST'])
def database():
    results = None
    resu = None
    conn = get_db()
    doi = request.form.get('doi', '').strip()
    
    if not doi:
        flash('Please enter a DOI', 'error')
    else:
        try:
                # Check retraction status
            retraction_df = pd.read_csv('retraction_watch.csv', dtype=str)
            retracted = doi in retraction_df['OriginalPaperDOI'].values
            reason = retraction_df.loc[retraction_df['OriginalPaperDOI'] == doi, 'Reason'].values[0] if retracted else ""
            reason_list = [item.strip() for item in reason.replace('+', '').split(';') if item.strip()]
            
            # Check Open access status
            oa = check_open_access(doi)
            print(f"OA data: {oa}")
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                AND name IN ('retraction')
            """)
            existing_tables = [row['name'] for row in cursor.fetchall()]
            
            if len(existing_tables) <= 0:
                flash(f"Missing tables. Found: {existing_tables}", 'error')
            else:
                cursor.execute("""
                    SELECT CitingDOI, CitationID, CCparagraph, Citation_context, Reason, RetractionDate,DOI, Citedtitle FROM retraction WHERE retraction.DOI = ?""", (doi,))
                results = cursor.fetchall()
                results_columns = [desc[0] for desc in cursor.description]
                print(results_columns)
                
                cursor.execute("""
                    SELECT * FROM retraction
                    WHERE retraction.CitingDOI = ?
                """, (doi,))
                resu = cursor.fetchall()
                #print(resu)
                resu_columns = [desc[0] for desc in cursor.description]
            
                if not results and not resu:
                    flash(f"No results found for DOI or CitingID: {doi}", 'info')

                results = [list(row) for row in results] if results else []
                resu = [list(row) for row in resu] if resu else []
                
                return render_template(
                    "db.html", 
                    results=results, 
                    resu=resu, 
                    doi=doi,
                    column_names=results_columns,
                    resu_column_names=resu_columns,
                    retracted=retracted,
                    reasons=reason_list,
                    oa=oa)
                            
        except sqlite3.Error as e:
            flash(f"Database error: {str(e)}", 'error')
        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
        finally:
            conn.close()

    #return redirect('/')

    
if __name__ == '__main__': 
    server.run(debug=True, threaded=True)
