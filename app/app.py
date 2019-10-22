import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

def load_scenario(i):
    path = os.path.abspath(f'./fake_data/scenario{i}.csv')
    data = pd.read_csv(path)
    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%S:%f'))
    return data
'''
Tokenization from scratch: some thoughts
 * train a character level language model. then iteratively join tokens such that the entropy of the predictions stays around constant
I wonder how predictive accuracy scales with the level of abstraction that someone is reasoning at.
'''
# LDD Region Drift Algorithm
def distance(a, b):
    return len(np.where(a != b))

def knn(centre, population, k):
    # Return the indices of the k-nearest neighbours of center in population
    distances = [ distance(centre, x) for x in population ]
    ret = []
    for dist in range(len(centre)):
        for i in np.where(distances == dist):
            ret.append(i)
            if len(ret) >= k:
                return ret
    #
    return ret

def get_ewma(x, alpha=0.05):
    # where x is some sequence
    ewma = [x[0]]
    for i, x_i in enumerate(x):
        if i==0:
            continue
        ewma.append( x_i * alpha + ewma[i-1] * (1-alpha) )
    return ewma

def get_graph(data, id):
    return dcc.Graph(
        id=id,
        figure={
            'data': data,
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )

def get_stream_plot(x, ys, id='', stacked=False):
    # ys is a dictionary {stream_name: stream_values}
    # if stack then stack all streams on one plot so that they sum to one
    ewmas = { name: get_ewma(y) for name, y in ys.items() }
    if stacked:
        data = [
            go.Scatter(x=x, y=ewma, mode='lines', stackgroup='one', name=name)
            for name, ewma in ewmas.items()
        ]
    else:
        data = [ { 'x': x, 'y': ewma, 'name': name } for name, ewma in ewmas.items() ]
    return get_graph(data, id)

def get_error_tab(data):
    ys = { col: data[col] for col in ['warning_level', 'drift_level', 'error_rate'] }
    return dcc.Tab(
        label='Error Rate',
        children=[
            html.Div(style={'backgroundColor': colors['background']}, children=[
                get_stream_plot(data['date'], ys, id='Graph1')
            ])
        ]
    )

def get_label_tab(data):
    ys = {i: data[f'label={i}'] for i in [1,2,3,4,5]}
    return dcc.Tab(
        label='Labels',
        children=[
        html.Div(style={'backgroundColor': colors['background']},
        children=[
                dcc.Checklist(
                    id = 'label display',
                    options=[
                        {'label': 'Predicted labels', 'value': 'predicted'},
                        {'label': 'True Labels', 'value': 'true'}
                    ],
                    value=['true']
                ),
                dcc.Checklist(
                    id = 'label metrics',
                    options=[
                        {'label': 'Precision', 'value': 'prec'},
                        {'label': 'Recall', 'value': 'rec'},
                        {'label': 'F1', 'value': 'f1'},
                        {'label': 'Frequency', 'value': 'freq'}
                    ],
                    value=['rec']
                ),
                dcc.Dropdown(
                    options=[
                        {'label': 'value', 'value': 'val'},
                        {'label': 'z-score', 'value': 'z'},
                        {'label': 'p-value', 'value': 'p'},
                        {'label': 'minus log p-value', 'value': 'log'}
                    ],
                    value='val'
                ),
                get_stream_plot(data['date'], ys, id='Graph2', stacked=True)
            ])
        ]
    )

def get_feature_tab(data, single_plot=False):
    feature_cols = [ col for col in data.columns if \
        col not in ['date'] + ['warning_level', 'drift_level', 'error_rate'] \
        and not col.startswith('label=') ]
    if single_plot:
        ys = {col: data[col] for col in feature_cols }
        children = [ get_stream_plot(data['date'], ys, id='FeaturesGraph') ]
    else:
        children = [ get_stream_plot(data['date'], {col: data[col]}, id=f'FeaturesGraph{i}') for i, col in enumerate(feature_cols[:50]) ]
    return dcc.Tab(
        label='Features',
        children=[
            html.Div(style={'backgroundColor': colors['background']}, children=children)
        ]
    )

def get_tabs(data):
    # return [ ErrorTab(data), LabelTab(data), FeatureTab(data) ]
    return [ get_error_tab(data), get_label_tab(data), get_feature_tab(data) ]

data = load_scenario(1)

app = dash.Dash()

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


colors = {
    'background': 'white',
    'text': 'black'
}

app.layout = html.Div([
    html.H1('GP Referrals Triage Drift Detection', style={
            'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'system-ui'}),
    # html.Label('Scenario',
    #             style={'fontFamily': 'system-ui', 'maxWidth': '1000px', 'margin': '10px auto'},
    # ),
    dcc.Dropdown(
        id = 'scenario-selector',
        options=[
            {'label': 'Real Drift', 'value': 1},
            {'label': 'Virtual Drift', 'value': 2}
        ],
        value=1,
        style={'fontFamily': 'system-ui', 'maxWidth': '1000px', 'margin': '10px auto'},
        # content_style={
        #     'borderLeft': '1px solid #d6d6d6',
        #     'borderRight': '1px solid #d6d6d6',
        #     'borderBottom': '1px solid #d6d6d6',
        #     'padding': '44px'
        # },

    ),
    dcc.Tabs(id="tabs", children=get_tabs(data),
        style={
            'fontFamily': 'system-ui'
        },
        content_style={
            'borderLeft': '1px solid #d6d6d6',
            'borderRight': '1px solid #d6d6d6',
            'borderBottom': '1px solid #d6d6d6',
            'padding': '44px'
        },
        parent_style={
            'maxWidth': '1000px',
            'margin': '0 auto'
        }
    )
])

@app.callback(
    Output(component_id='tabs', component_property='children'),
    [Input(component_id='scenario-selector', component_property='value')]
)
def update_output_div(input_value):
    global data
    data = load_scenario(input_value)
    return get_tabs(data)
    # return 'You\'ve entered "{}"'.format(input_value)


if __name__ == '__main__':
    app.run_server(debug=True)
