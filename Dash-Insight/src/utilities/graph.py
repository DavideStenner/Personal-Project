import plotly.graph_objects as go
import plotly.express as px

# Generical blank figure before initialization
def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig

# Generical scatter plot with template
def get_scatter(data, x, y, color = None, template = 'plotly_white'):
    fig = px.scatter(
        data, x=x, y=y, color=color,
        template = template,
        labels = {i: i.capitalize() for i in [x, y]}
    )

    return fig

# Histogram scatter plot with template
def get_histogram(data, x, color = None, template = 'plotly_white'):
    fig = px.histogram(
        data, x=x, color=color,
        template = template,
        labels = {x: x.capitalize()},
    )

    return fig
