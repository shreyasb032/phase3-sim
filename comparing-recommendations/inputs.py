from dash import dcc, html


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
                           value=90,
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
                         value=90,
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


sliders = html.Div(children=[
    threat_level_slider_div,
    html.Br(),
    health_slider_div,
    html.Br(),
    time_slider_div,
    html.Br(),
    wh_slider_div
], className='Inputs', style={'padding': 20, 'flex': 1})


threat_level_input = html.Div(children=[
    "Prior Threat Level: ",
    dcc.Input(value=0.7, min=0.0, max=1.0, step=0.05, type='number', id='prior-threat-input')
], style={'padding': 20})

health_input = html.Div(children=[
    "Health remaining: ",
    dcc.Input(value=100, min=0, max=100, step=10, type='number', id='health-input')
], style={'padding': 20})

time_input = html.Div(children=[
    "Time remaining: ",
    dcc.Input(value=100, min=0, max=100, step=10, type='number', id='time-input')
], style={'padding': 20})

wh_input = html.Div(children=[
    "wh (constant): ",
    dcc.Input(value=0.75, min=0.0, max=1.0, step=0.01, type='number', id='wh-constant-input')
], style={'padding': 20})

text_inputs = html.Div([
    threat_level_input,
    html.Br(),
    health_input,
    html.Br(),
    time_input,
    html.Br(),
    wh_input
], className='TextInputs', style={'padding': 20, 'flex': 1})
