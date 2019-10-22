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
The authors describe politics in terms of three dimensions: essentials, influentials, and interchangeables. Essentials are people the political leader needs to keep loyal, influentials are people who the leader sometimes needs to keep loyal, and interchangeables are people who the leader doesn't need to keep loyal on an individual basis. This framework seems kinda strange to me. These seem more like quantitative distinctions than qualititative ones, so I think it would be more natural to put everyone on a single axis of "influence" or "value to the leader". This way there would be an easy way to quantify where a given society falls on the democracy-autocracy spectrum, by taking either the entropy, or the power law index of the "influence" distribution. I don't see an easy way to do this with the essentials/influentials/interchangeables framework (is lots of interchangeables but few influentials better than vice versa?)
On foreign aid:
  * State foreign aid seems to be motivated by the interests of the populous rather than humanitarian interests. The most eggregious example of this was the US personally enriching Liberia's dictator Samuel Doe in exchange for his loyalty against communism.
  * Something I've found confusing is that countries rich in natural resources often do worse in terms of development than those poor in natural resources. It's easy to see why they wouldn't do any better: a dictator simply steals all the value of the natural resources and so none goes into development. But why would these countries actually do <i>worse</i> in development? There are two reasons: if the dictator has more money readily available then they can more successfully supress any political dissidents. And, more interestingly, if there are no natural resources to plunder, the only way for the dictator to enrich themself and pay off their cronies is by taxing a productive population. Thus economic development is in the dictator's interest.
  * The best way to help a poor country under autocratic rule is 1) don't buy natural resources from it as this only empowers the dictator (see above), 2) reward the dictator for taking steps towards democratization (such as offering safe and luxurious retirements to dictators who agree to step down), 3) don't give the rewards directly, instead place them in escrow accounts which will be opened after certain stipulations are met.
  * Humanitarian efforts are typically most successful in areas which the local dictator would support anyway. Health, nutrition, and low-level education are all capabilities which make the population more productive, and therefore more wealth-extractable.
  * Bringing aid or forgiving debts of corrupt and poor countries is often counterproductive, as this only gives the dictator more opportunities to pay off his cronies and entrench his position.
War:
  * In democracies, the value of citizen lives is higher, and if the security of the state is at risk, then
  * Why is Israel so milataristically successful? Partly
That democrats are more
A consequence of gerrymandering is that everyone is dissatisfied with the government as a whole, but satisfied with their representitive in particular.


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
