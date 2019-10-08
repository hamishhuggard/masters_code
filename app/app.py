import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import json
import os

data_streams_path = os.path.abspath('./fake_data/data_streams.json')

with open(data_streams_path) as f:
    data_streams = json.loads(f.read())

x = list(range(len(data_streams['accuracy'])))

app = dash.Dash()
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='GP Referrals Triage Drift Detection',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children='Below is a time series of the error rate of the model and the thresholds.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(
        id='Graph1',
        figure={
            'data': [
                {'x': stream[0], 'y': stream[1], 'name': name}
                for name, stream in data_streams.items()
#                {'x': x, 'y': warnings, 'name': 'Warning Threshold'}, # 'type': 'bar',##
#                {'x': x, 'y': drifts, 'name': 'Drift Threshold'},
#                {'x': x, 'y': errs, 'name': u'Error rate'},
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

if __name__ == '__main__':
    app.run_server(debug=True)
