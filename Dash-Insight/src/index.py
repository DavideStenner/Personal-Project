import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

from app import app
from apps import app1, app2, main_page, upload_file
from configuration.constant import SIDEBAR_STYLE, CONTENT_STYLE

first_page = html.Div(
    id = 'main_page'
)

sidebar = html.Div(
    [
        dbc.Nav(
            [
                dbc.NavLink("Summary", href="/", active="exact"),
                dbc.NavLink('Upload File', href="/apps/upload_file", active="exact")
            ],
            vertical=True,
            pills=True,
        )
    ],
    style=SIDEBAR_STYLE,
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    sidebar,
    html.Div(id='page-content', style=CONTENT_STYLE),
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return main_page.layout
    elif pathname == '/apps/upload_file':
        return upload_file.layout
    else:
        return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=True)
