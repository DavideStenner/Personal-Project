import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_html_components import H6
import pandas as pd
from dash_table import DataTable
import plotly.graph_objects as go
import numpy as np
from app import app
from utilities.import_data import import_dataset, import_metadata
from utilities.graph import blank_fig

figure_table = html.Div([
    dcc.Graph(
        id='summary-table',
        figure = blank_fig()
    )
])

figure_hist = html.Div([
    dcc.Graph(
            id='barplot-figure',
            figure = blank_fig()
    )
])

dropdown_container = html.Div(
    [
        'Select Categorical Column',
        dcc.Dropdown(
            id='dropdown-string-value',
            options=[],
            clearable=False
        )
    ]
)

table = html.Div([
    DataTable(
        id='datatable-interactivity',
        columns=None,
        data=None,
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
        },
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
                    html.Br()
                ],
                style={'width': '75%', 'display': 'inline-block'}
            ),
            html.Div(
                [
                   dropdown_container 
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
        html.Div([
            html.H6('Summary Table'),
            figure_table
            ], 
            style={'width': '40%', 'display': 'inline-block'}
        ),
        html.Div([
            html.H6('Histogram'),
            figure_hist
            ], 
            style={'width': '40%', 'display': 'inline-block'}
        ),
    ]),
])

#update the dataset
@app.callback(
    [   
        Output('datatable-interactivity', 'columns'),
        Output('datatable-interactivity', 'data')],
    Input('url', "pathname"))
def update_dataset(url):
    
    df = import_dataset()

    #need to import something
    if df is None:
        return [None, None]

    columns = [
        {"name": i, "id": i} for i in df.columns
    ]

    return [columns, df.to_dict('records')]

#Update number of rows
@app.callback(
    Output('information-dataframe-container', "children"),
    [
        Input('datatable-interactivity', "derived_virtual_data"),
        Input('url', "pathname")
    ])
def update_info_rows(rows, url):

    df = import_dataset()
    
    #need to import something
    if df is None:
        return [None]

    dff = df if rows is None else pd.DataFrame(rows)
    num_rows, num_cols = dff.shape

    return [
        html.P(f'The dataset has {num_rows} rows and {num_cols} columns.')
    ]

#Update dropdown for string
@app.callback(
        Output('dropdown-string-value', "options"),
        Input('url', "pathname"))
def update_dropdown(url):

    _, metadata_dic = import_metadata(get_type_colname = True)

    #need to import something
    if metadata_dic is None:
        return []

    options = [
        {'label': col, 'value': col}
        for col in metadata_dic['string']
    ]

    return options


#Updates Bar plot
@app.callback(
    Output('barplot-figure', "figure"),
    Input('dropdown-string-value', 'value'),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('url', "pathname"))
def update_graphs(string_col, rows, url):

    if string_col is None:
        return blank_fig()

    df = import_dataset()

    dff = df if rows is None else pd.DataFrame(rows)

    # if you filter and have empty dataframe
    if (string_col not in dff.columns):
        return blank_fig()

    #coherce to string... integer will be displayed as str
    dff_string_unique = [str(x) for x in dff[string_col].unique().tolist()]
    dff_string_count = [(dff[string_col] == cat).sum() for cat in dff[string_col].unique()]

    fig = go.Figure(
        data = go.Bar(
            x = dff_string_unique,
            y = dff_string_count
        ),
        layout = go.Layout(template = 'plotly_white', title = string_col, margin={'l': 0})
    )

    return fig

#Updates summary table
@app.callback(
    Output('summary-table', "figure"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('url', "pathname"))
def update_table(rows, url):
    
    _, metadata_type = import_metadata(get_type_colname = True)

    df = import_dataset()

    #need to import something
    if (df is None) or (metadata_type is None):
        return blank_fig()


    dff = df if rows is None else pd.DataFrame(rows)

    numeric_col = [x for x in metadata_type['numeric'] if x in dff.columns]

    numeric_info = [
        [
            col,
            np.round(dff[col].mean(), 2),
            np.round(dff[col].std(), 2),
            np.round(dff[col].median(), 2),
            np.round(dff[col].min(), 2),
            np.round(dff[col].max(), 2),
        ] for col in numeric_col
    ]

    #reshape
    numeric_info = np.array(numeric_info).T.tolist()

    fig_table = go.Figure(
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

    return fig_table