from os import stat
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
from dash_table import DataTable
import json
import plotly.graph_objects as go
import numpy as np
from app import app
from configuration.constant import NUMERIC_COL, STRING_COL, DF

df = pd.read_csv('./dataset/dataset.csv')

figure_table = html.Div([
    html.Div(id='summary-table-container')
])

figure_hist = html.Div([
    html.Div(id='hist-container')
])

dropdown_element = dcc.Dropdown(
                        id='dropdown-string-value',
                        options=[
                            {'label': col, 'value': col}
                            for col in STRING_COL
                        ],
                        value=STRING_COL[0],
                        clearable=False
                    )
table = html.Div([
    DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i} for i in DF.columns
        ],
        data=DF.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable=False,
        row_deletable=False,
        page_action="native",
        page_current= 0,
        page_size= 10,
        column_selectable="single",
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    ),
    html.Div(id='datatable-interactivity-container')
])

information_df = html.Div([
    html.Div(id = 'information-dataframe-container')
])

layout = html.Div([
    html.Div(
        [
            html.Div(
                [
                    html.H3('Dataset Statistics'),
                    html.Br(),
                ],
                style={'width': '75%', 'display': 'inline-block'}
            ),
            html.Div(
                [
                   dropdown_element 
                ],
                style={'width': '24%', 'display': 'inline-block'}
            )
    ]),
    html.Div([
        html.H5('Dataset'),
        html.Br(),
        information_df,
        table
    ]),
    html.Div([
        html.Br(),
        html.H5('Summary Table'),
        html.Div([
            figure_table
            ], 
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div([
            figure_hist
            ], 
            style={'width': '49%', 'display': 'inline-block'}
        ),
    ]),
])

#Update number of rows
@app.callback(
    Output('information-dataframe-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"))
def update_info_rows(rows):
    dff = DF if rows is None else pd.DataFrame(rows)
    num_rows, num_cols = dff.shape

    return [
        html.P(f'The dataset has {num_rows} rows and {num_cols} columns.')
    ]

#Updates summary table
@app.callback(
    Output('hist-container', "children"),
    Input('dropdown-string-value', 'value'),
    Input('datatable-interactivity', "derived_virtual_data"))
def update_graphs(string_col, rows):

    dff = DF if rows is None else pd.DataFrame(rows)

    #coherce to string... integer will be displayed as str
    dff_string_unique = [str(x) for x in dff[string_col].unique().tolist()]
    dff_string_count = [(dff[string_col] == cat).sum() for cat in dff[string_col].unique()]

    fig_barplot = [
        dcc.Graph(
            id=string_col,
            figure = go.Figure(
                data = go.Bar(
                    x = dff_string_unique,
                    y = dff_string_count
                ),
                layout = go.Layout(template = 'plotly_white', title = string_col, margin={'l': 0})
            )
        )
    ]

    return fig_barplot

#Updates summary table
@app.callback(
    Output('summary-table-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"))
def update_graphs(rows):
    
    dff = DF if rows is None else pd.DataFrame(rows)
    numeric_info = [
        [
            col,
            np.round(dff[col].mean(), 2),
            np.round(dff[col].std(), 2),
            np.round(dff[col].median(), 2),
            np.round(dff[col].min(), 2),
            np.round(dff[col].max(), 2),
        ] for col in NUMERIC_COL
    ]

    #reshape
    numeric_info = np.array(numeric_info).T.tolist()

    fig_table = dcc.Graph(
        id='summary-table',
        figure = go.Figure(
            data=[
                go.Table(
                    header=dict(values=['Columns', 'Mean', 'Std', 'Median', 'Min', 'Max'],
                        align=['left','center']
                    ),
                    cells=dict(
                        values=numeric_info,
                        align=['left', 'center']
                    )
                )
            ],
            layout = go.Layout(template = 'plotly_white', margin={'l': 0})
        )
    )

    return [
        fig_table
    ]