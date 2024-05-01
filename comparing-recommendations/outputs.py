from dash import html, dcc


d_star_1_div = html.Div(children=[
    html.Div(r'$d_1^*$: Value', id='d-1-star'),
    # html.Div('Value')
])

d_star_2_div = html.Div(children=[
    html.Div(children=[
        html.Div(r"$d_2^*$: Value", id='d-2-star'),
        # html.Div('Value')
    ])
])

d_star_div = html.Div(children=[
    d_star_1_div,
    html.Br(),
    d_star_2_div
], style={'width': 300, 'padding': 20, 'padding-left': 400, 'flex': 1})

d_hat_div = html.Div(children=[
    html.Div(children=[
        r"\hat{d}: ",
        dcc.Input(value=0.7, type="number", min=0.0, max=1.0, step=0.01, id='d-hat-input'),
    ])
], style={'flex': 1, 'padding': 20})

recommendation_1_div = html.Div(children=[
    html.Div(r'state dependent recommendation: Value', id='state-rec'),
    # html.Div('Value')
])

recommendation_2_div = html.Div(children=[
    html.Div(children=[
        html.Div(r"constant recommendation: Value", id='constant-rec'),
        # html.Div('Value')
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
