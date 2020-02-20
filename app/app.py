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
    # path = os.path.abspath(f'./fake_data/scenario{i}.csv')
    # data = pd.read_csv(path)
    # data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%S:%f'))
    dirname = os.path.abspath(f'./fake_data/scenarios/{i}/')
    data = pd.read_csv(dirname + '/test.csv')
    with open(dirname + '/y.json', 'r') as f:
        y = json.loads(f.read())
    with open(dirname + '/errs.json', 'r') as f:
        errs = json.loads(f.read())
    err_data = pd.read_csv(dirname + '/err_data.csv')
    status_data = pd.read_csv(dirname + '/status_data.csv')
    for i in [1,2,3,4,5]:
        data[f'label={i}'] = [ label==i for label in y ]
    data['error_rate'] = errs
    data['date'] = list(range(len(data)))
    # data['err_data'] = err_data
    data = data.join(err_data)
    data = data.join(status_data)
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

# def

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
                },
                'title': id
            }
        }
    )

def get_stream_plot(x, ys, id='', stacked=False, color=[]):
    # ys is a dictionary {stream_name: stream_values}
    # if stack then stack all streams on one plot so that they sum to one
    ewmas = { name: get_ewma(y) for name, y in ys.items() }
    args = {'x': x, 'mode': 'lines'}
    if stacked:
        args['stackgroup'] = 'one'
    if len(color) > 0:
        args['marker'] = {'color': color, 'colorscale': 'rdylgn'}#, 'showscale': True}
        args['mode'] = 'lines+markers'
        args['line'] = {'color': 'black'}
    data = []
    for name, ewma in ewmas.items():
        args['y'] = ewma
        args['name'] = name
        data.append( go.Scatter(**args) )
    # if len(color) > 0:
    #     args['mode'] = 'marker'
    #     for name, ewma in ewmas.items():
    #         args['y'] = ewma
    #         args['name'] = name
    #         data.append( go.Scatter(**args) )
    # data = [ { 'x': x, 'y': ewma, 'name': name } for name, ewma in ewmas.items() ]
    # if stacked:
    #
    #     data = [
    #         go.Scatter(x=x, y=ewma, color=color, mode='lines', stackgroup='one', name=name)
    #         for name, ewma in ewmas.items()
    #     ]
    # else:
    #     data = [ { 'x': x, 'y': ewma, 'name': name } for name, ewma in ewmas.items() ]
    return get_graph(data, id)

def get_error_tab(data):
    ys = { col: data[col] for col in ['error_rate'] } # 'warning_level', 'drift_level',
    status = data['STATUS'].astype('category').cat.set_categories(['Drift', 'Warning', 'Normal'])
    print(status.cat.categories)
    status = status.cat.reorder_categories(['Drift', 'Warning', 'Normal'])

    return dcc.Tab(
        label='Error Rate',
        children=[
            html.Div(style={'backgroundColor': colors['background']}, children=[
                get_stream_plot(data['date'], ys, id='Model Error Rate', color=status.cat.codes)
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
                # dcc.Checklist(
                #     id = 'label display',
                #     options=[
                #         {'label': 'Predicted labels', 'value': 'predicted'},
                #         {'label': 'True Labels', 'value': 'true'}
                #     ],
                #     value=['true']
                # ),
                # dcc.Checklist(
                #     id = 'label metrics',
                #     options=[
                #         {'label': 'Precision', 'value': 'prec'},
                #         {'label': 'Recall', 'value': 'rec'},
                #         {'label': 'F1', 'value': 'f1'},
                #         {'label': 'Frequency', 'value': 'freq'}
                #     ],
                #     value=['rec']
                # ),
                # dcc.Dropdown(
                #     options=[
                #         {'label': 'value', 'value': 'val'},
                #         {'label': 'z-score', 'value': 'z'},
                #         {'label': 'p-value', 'value': 'p'},
                #         {'label': 'minus log p-value', 'value': 'log'}
                #     ],
                #     value='val'
                # ),
                get_stream_plot(data['date'], ys, id='Label Rates', stacked=True)
                ] + [ get_stream_plot(data['date'], {f'label={i}': data[f'label={i}']}, id=f'Label {i} Rate') for i in range(1, 6)
            ])
        ]
    )

def get_feature_tab(data, single_plot=False):
    feature_cols = [ col for col in data.columns if \
        col not in ['date'] + ['warning_level', 'drift_level', 'error_rate'] + \
        ['STATUS', 'ERRS', 'PROB'] \
        and not col.startswith('label=') and not col.endswith('_status')]
    if single_plot:
        ys = {col: data[col] for col in feature_cols }
        children = [ get_stream_plot(data['date'], ys, id='FeaturesGraph') ]
    else:
        children = [ get_stream_plot(data['date'], {col: data[col]}, id=f'Rate of Feature "{col}"', color=data[f'{col}_status']) for i, col in enumerate(feature_cols[:50]) ]
    return dcc.Tab(
        label='Features',
        children=[
            html.Div(style={'backgroundColor': colors['background']}, children=children)
        ]
    )

def get_tabs(data):
    # return [ ErrorTab(data), LabelTab(data), FeatureTab(data) ]
    return [ get_error_tab(data), get_label_tab(data), get_feature_tab(data) ]

data = load_scenario('shuffle_features') # 1

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
            # {'label': 'Real Drift', 'value': 1},
            # {'label': 'Virtual Drift', 'value': 2}
            {'label': scenario, 'value': scenario}
            for scenario in ['shuffle_features', 'not_feature', 'swap_classes', 'new_concept']
        ],
        value='shuffle_features', # 1
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
