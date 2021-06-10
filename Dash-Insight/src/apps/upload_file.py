import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
from app import app
import base64
import io
import time
import numpy as np

layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            dbc.Alert([
                    'Drag and Drop or ',
                    html.A('Select Files', href="#", className="upload-link")
                ], 
                color="primary",
            )
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Br(),
    html.Br(),
    html.Div(id='output-data-upload'),
])


def parse_contents(contents, filename, date):
    start_time = time.time()

    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        if ('csv' in filename) or ('txt' in filename):
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise IOError

    except IOError as e:

        return html.Div([
            'There was an error processing this file:   ', filename
        ])

    if 'metadata' in filename:
        df.to_excel('./dataset/metadata.xlsx', index = False)
    else:
        df.to_csv('./dataset/dataset.csv', index = False)

    additional_string = f'\nNumber of rows and columns uploaded: {df.shape}.' if 'metadata' not in filename else ''
    end_time = time.time()

    range_time = np.round(end_time - start_time, 2)
    return html.Div([
        html.H5(filename),
        html.Pre(f'Time to upload: {range_time} ' + additional_string),
    ])


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        if len(list_of_names) > 2:
            return html.Div([
                html.Pre(f'Too many file uploaded. The application need a dataset and a metadata file with information about type of each columns.'),
            ])
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children
