import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from dash_table import DataTable
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from app import app
from utilities.import_data import import_dataset, import_metadata
from utilities.graph import blank_fig

figure_scatter = html.Div([
    dcc.Graph(
        id='numeric-multi-figure',
        figure = blank_fig()
    )
])

figure_hist = html.Div([
    dcc.Graph(
            id='categorical-multi-figure',
            figure = blank_fig()
    )
])

dropdown_container_scatter = html.Div(
    html.Div([
        'Select Columns',
        dcc.Dropdown(
            id='dropdown-scatterplot-value',
            options=[],
            multi=True,
            clearable=False,
        )
        ],
        style={'width': '35%', 'display': 'inline-block'}
    )
)

dropdown_container_hist = html.Div(
    html.Div([
        'Select Categorical Columns',
        dcc.Dropdown(
            id='dropdown-hist-value',
            options=[],
            clearable=False,
            multi = False,
        )
        ],
        style={'width': '35%', 'display': 'inline-block'}
    )
)

hist_container = html.Div(
    [
        dropdown_container_hist,
        html.Br(),
        figure_hist
    ], 
    style={'width': '50%', 'display': 'inline-block'}
)

scatter_container = html.Div(
    [
        dropdown_container_scatter,
        html.Br(),
        figure_scatter
    ], 
    style={'width': '50%', 'display': 'inline-block'}
)

layout = html.Div(
    [
        scatter_container, 
        hist_container
    ]
)

#Update dropdown selection for axis
@app.callback(
        Output('dropdown-scatterplot-value', "options"),
        Input('dropdown-scatterplot-value', 'value'),
        Input('url', "pathname"))
def update_scatter_dropdown(value, _):

    _, metadata_dic = import_metadata(get_type_colname = True)

    #need to import something
    if metadata_dic is None:
        return True, []

    value_selected = [] if value is None else value

    options = [
        {'label': col, 'value': col} for col in metadata_dic['numeric']
    ]

    if len(value_selected) == 2:
        options = [
            {'label': col, 'value': col, 'disabled': True} if col not in value_selected else {'label': col, 'value': col}
            for col in metadata_dic['numeric']
        ]

    return options

# #Update dropdown selection for hist
@app.callback(
        Output('dropdown-hist-value', "options"),
        Input('url', "pathname"))
def update_scatter_dropdown(_):

    _, metadata_dic = import_metadata(get_type_colname = True)

    #need to import something
    if metadata_dic is None:
        return []

    options = [
        {'label': col, 'value': col} for col in metadata_dic['string']
    ]

    return options


#Updates scatter plot
@app.callback(
    Output('numeric-multi-figure', "figure"),
    Input('dropdown-scatterplot-value', "value"),
    Input('url', "pathname"))
def update_scatter(selection, _):

    #blank before selection
    if (selection is None) or (selection == []):
        return blank_fig()

    df = import_dataset()

    #assign x
    x_selection = selection[0]

    #return histogram
    if (len(selection) == 1):
        
        fig = px.histogram(
            df, x=x_selection, #color="Fine Corso",
            template = 'plotly_white',
            labels = {x_selection: x_selection.capitalize()},
        )
    else:
        y_selection = selection[1]
        fig = px.scatter(
            df, x=x_selection, y=y_selection, #color="species",
            template = 'plotly_white',
            # size='petal_length', hover_data=['petal_width']
        )



    return fig

#Updates hist plot
@app.callback(
    Output('categorical-multi-figure', "figure"),
    Input('dropdown-hist-value', "value"),
    Input('url', "pathname"))
def update_hist(selection, _):

    #blank before selection
    if (selection is None) or (selection == []):
        return blank_fig()

    df = import_dataset()

    df[selection] = df[selection].astype(str)

    fig = px.histogram(
        df, x = selection,
        template = 'plotly_white',
        labels = {selection: selection.capitalize()},
    )

    return fig
