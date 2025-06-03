from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import sqlite3
import csv
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'

BASE_DIR = "../" #os.path.dirname(os.path.abspath(__file__)) # current directory
db_path = os.path.join(BASE_DIR, "database")

def get_db():
    """Ensure database exists and get connection"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found.")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/db', methods=['GET', 'POST'])
def index():
    results = None
    conn = get_db()
    if request.method == 'POST':
        doi = request.form.get('doi', '').strip()
        if not doi:
            flash('Please enter a DOI', 'error')
        else:
            try:
                cursor = conn.cursor()
                
                # Verify both tables exist
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
                        SELECT retraction.*
                        WHERE retraction.DOI = ?
                    """, (doi, doi))
                    
                    results = cursor.fetchall()
                    
                    if not results:
                        flash(f"No results found for DOI: {doi}", 'info')
                    else:
                        # Save results to CSV
                        filename = f"results_{doi.replace('/', '_')}.csv"
                        with open(filename, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([col[0] for col in cursor.description])
                            writer.writerows([list(row) for row in results])
                        
            except sqlite3.Error as e:
                flash(f"Database error: {str(e)}", 'error')
            except Exception as e:
                flash(f"Error: {str(e)}", 'error')
            finally:
                conn.close()
    
    return render_template('db.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
                        
    '''# Execute the join query
        cursor.execute("""
            SELECT cit.*, retraction_watch.Reason 
            FROM cit
            INNER JOIN retraction_watch 
            ON cit.DOI = retraction_watch.OriginalPaperDOI
            WHERE cit.DOI = ? OR retraction_watch.OriginalPaperDOI = ?
        """, (doi, doi))'''
