import sys
import time
import requests
from cc import *
import pandas as pd
from flask import *
#from gevent import monkey
from fileinput import filename 
from distutils.log import debug 
import xml.etree.ElementTree as ET
#from gevent.pywsgi import WSGIServer
from concurrent.futures import ThreadPoolExecutor

#monkey.patch_socket()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():    
    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        result_df = extract_pmc_citations(f)
        print(result_df)
        return render_template("index.html", filename=filename, result_df=result_df,column_names=result_df.columns.values,                  row_data=list(result_df.values.tolist()),  zip=zip)
    return render_template("index.html", result_df=None)
    
if __name__ == '__main__': 
    app.run(debug=True, threaded=True)

