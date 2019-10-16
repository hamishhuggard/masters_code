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

def get_error_tab(data):
    return dcc.Tab(
        label='Error Rate',
        children=[
            html.Div(style={'backgroundColor': colors['background']}, children=[
                dcc.Graph(
                    id='Graph1',
                    figure={
                        'data': [
                            {'x': data['date'], 'y': data[col], 'name': col}
                            for col in ['warning_level', 'drift_level', 'error_rate']
                        ],
                        'layout': {
                            'plot_bgcolor': colors['background'],
                            'paper_bgcolor': colors['background'],
                            'font': {
                                'color': colors['text']
                            }
                        }
                    }
                )
            ])
        ]
    )

def get_label_tab(data):

    # x = data.date
    # fig = go.Figure()
    #
    # # colors = [ '184, 247, 212',  '111, 231, 219', '127, 166, 238', '131, 90, 241' ]
    #
    # for i in range(5):
    #     fig.add_trace(go.Scatter(
    #         x=x, y=data[f'label={i+1}'],
    #         mode='lines',
    #         line=dict(width=0.5),#, color=f'rgb({colors[i]})'),
    #         stackgroup='one'#,
    #         # groupnorm='percent' # sets the normalization for the sum of the stackgroup
    #     ))

    # fig.update_layout(
    #     showlegend=True)#,
        # xaxis_type='category',
        # yaxis=dict(
        #     type='linear',
        #     range=[1, 100],
        #     ticksuffix='%'))
    # label_colours = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'blue', 5: 'green'}
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
                dcc.Graph(
                    id='Graph2',
                    figure={
                        'data': [
                            # fig
                            # {'x': data['date'], 'y': streams[i], 'name': i} # data[f'label={i}']
                            # for i in [1,2,3,4,5]
                            go.Scatter(
                                x=data['date'], y=data[f'label={i}'], mode='lines', stackgroup='one', name=i#, fillcolor=label_colours[i]
                            )
                            for i in [1,2,3,4,5]
                        ],
                        'layout': {
                            'plot_bgcolor': colors['background'],
                            'paper_bgcolor': colors['background'],
                            'font': {
                                'color': colors['text']
                            }
                        }
                    }
                )
            ])
        ]
    )

def get_feature_tab(data):
    return dcc.Tab(
        label='Features',
        children=[
            html.Div(style={'backgroundColor': colors['background']}, children=[
                dcc.Graph(
                    id='Graph3',
                    figure={
                        'data': [
                            {'x': data['date'], 'y': data[col], 'name': col}
                            for col in data.columns if col != 'date' and \
                            not col.startswith('label=') and col not in ['warning_level', 'drift_level', 'error_rate']
                        ],
                        'layout': {
                            'plot_bgcolor': colors['background'],
                            'paper_bgcolor': colors['background'],
                            'font': {
                                'color': colors['text']
                            }
                        }
                    }
                )
            ])
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
