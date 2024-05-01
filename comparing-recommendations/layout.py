# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, html, dcc
import plotly.express as px
import pickle

app = Dash(__name__)

threat_level_slider = dcc.Slider(min=0.0, max=1.0, step=0.05,
                                 marks={
                                     0: '0',
                                     0.2: '0.2',
                                     0.4: '0.4',
                                     0.6: '0.6',
                                     0.8: '0.8',
                                     1.0: '1.0'
                                 },
                                 value=0.7,
                                 id='threat-level-slider',
                                 tooltip={"placement": "bottom", "always_visible": True})

threat_level_slider_div = html.Div(children=[
    html.Label('Prior threat level', style={'padding': 20}),
    threat_level_slider
], style={'padding-bottom': 20, 'padding-top': 80})

health_slider = dcc.Slider(min=10, max=100, step=10,
                           marks={
                               10: '10',
                               30: '30',
                               50: '50',
                               70: '70',
                               90: '90',
                               100: '100'
                           },
                           value=100,
                           id='health-slider',
                           tooltip={"placement": "bottom", "always_visible": True})

health_slider_div = html.Div(children=[
    html.Label('Health remaining', style={'padding': 20}),
    health_slider
], style={'padding-bottom': 20})

time_slider = dcc.Slider(min=10, max=100, step=10,
                         marks={
                             10: '10',
                             30: '30',
                             50: '50',
                             70: '70',
                             90: '90',
                             100: '100'
                         },
                         value=100,
                         id='time-slider',
                         tooltip={"placement": "bottom", "always_visible": True})

time_slider_div = html.Div(children=[
    html.Label('Time remaining', style={'padding': 20}),
    time_slider
], style={'padding-bottom': 20})


wh_slider = dcc.Slider(min=0.0, max=1.0, step=0.05,
                       marks={
                           0: '0',
                           0.2: '0.2',
                           0.4: '0.4',
                           0.6: '0.6',
                           0.8: '0.8',
                           1.0: '1.0'
                       },
                       value=0.75,
                       id='wh-slider',
                       tooltip={"placement": "bottom", "always_visible": True})

wh_slider_div = html.Div(children=[
    html.Label('wh (constant)', style={'padding': 20}),
    wh_slider
], style={'padding-bottom': 20})


inputs = html.Div(children=[
    threat_level_slider_div,
    html.Br(),
    health_slider_div,
    html.Br(),
    time_slider_div,
    html.Br(),
    wh_slider_div
], className='Inputs', style={'padding': 20, 'flex': 1})

with open('../models/weights_for_plot.pkl', 'rb') as f:
    wh_pred = pickle.load(f)

fig = px.imshow(wh_pred, color_continuous_scale='magma', zmin=0.5, zmax=1.0, origin='lower',
                width=500, height=500, labels={'x': 'Time remaining', 'y': 'Health remaining'})
graph = dcc.Graph(id='state-dependent-weights',
                  figure=fig)

row_break = html.Div(children=None,
                     className='break',
                     style={'flex-basis': '100%', 'height': 0})

d_star_1_div = html.Div(children=[
    html.Div(r'$d_1^*$'),
    html.Div('Value')
])

d_star_2_div = html.Div(children=[
    html.Div(children=[
        html.Div(r"$d_2^*$"),
        html.Div('Value')
    ])
])

d_star_div = html.Div(children=[
    d_star_1_div,
    html.Br(),
    d_star_2_div
], style={'width': 300, 'padding': 20, 'padding-left': 400, 'flex': 1})

d_hat_div = html.Div(children=[
    html.Div(children=[
        html.Div(r"\hat{d}"),
        html.Div("Value")
    ])
], style={'flex': 1, 'padding': 20})

recommendation_1_div = html.Div(children=[
    html.Div(r'state dependent robot recommendation'),
    html.Div('Value')
])

recommendation_2_div = html.Div(children=[
    html.Div(children=[
        html.Div(r"constant robot recommendation"),
        html.Div('Value')
    ])
])

recommendation_div = html.Div(children=[
    recommendation_1_div,
    html.Br(),
    recommendation_2_div
], style={'width': 500, 'padding': 20, 'flex': 1})


outputs = html.Div(children=[
    d_star_div,
    d_hat_div,
    recommendation_div,
], style={'padding': 20, 'display': 'flex'})

app.layout = html.Div([
    inputs,
    html.Div(children=[
        html.Label('State Dependent Weights'),
        graph
    ], className='WeightsGraph', style={'padding': 20, 'flex': 1}),
    row_break,
    outputs
], style={'display': 'flex', 'flexDirection': 'row', 'flex-wrap': 'wrap'})

if __name__ == '__main__':
    app.run(debug=True)
