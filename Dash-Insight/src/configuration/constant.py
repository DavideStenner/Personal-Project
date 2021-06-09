import pandas as pd

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

####
##DATASET
DF = pd.read_csv('./dataset/dataset.csv')
METADATA = pd.read_excel('./dataset/metadata.xlsx')

NUMERIC_COL = METADATA.loc[METADATA['type'] == 'number', 'col_name'].tolist()
STRING_COL = METADATA.loc[~METADATA['type'].isin(['number', 'date']), 'col_name'].tolist()
DATE_COL = METADATA.loc[METADATA['type'] == 'date', 'col_name'].tolist()
