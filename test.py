import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

app = dash.Dash(__name__)
server = app.server 

df = pd.read_csv('../Retraction/retraction_watch.csv', dtype=str)

# First convert to datetime properly
df['RetractionDate'] = pd.to_datetime(
    df['RetractionDate'],
    format='%m/%d/%Y %H:%M', 
    errors='coerce'        
)

df['Year'] = df['RetractionDate'].dt.year

# Drop rows with invalid dates (if any)
df = df.dropna(subset=['Year'])
columns = ['Subject','Journal','Publisher','Year']

app.layout = html.Div([
    html.H3("Retraction watch stats"),
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
            color_discrete_sequence=['#fc1616']
        )
    
    elif 'year' in selected_column.lower() and pd.api.types.is_numeric_dtype(df[selected_column]):
        filtered_df = df[df[selected_column] >= 2008].copy()  
        filtered_df.loc[:, selected_column] = filtered_df[selected_column].astype(int).astype(str)
        
        fig = px.histogram(
            filtered_df,
            x=selected_column,
            color_discrete_sequence=['#fc1616'],
            category_orders={selected_column: sorted(filtered_df[selected_column].unique())}
        )
    
    else:
        fig = px.histogram(
            df, 
            x=selected_column, 
            nbins=25,
            color_discrete_sequence=['#fc1616']
        )

        hoverlabel={
            'font': {'color': 'black'}
        },
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
    bargap=0.2 )
    return fig

if __name__ == '__main__':
    app.run(debug=True)

