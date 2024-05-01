# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, Input, Output, callback
from inputs import sliders, text_inputs
from graph import graph, wh_predictions
from row_break import row_break
from outputs import outputs
import plotly.express as px


app = Dash(__name__)

app.layout = html.Div([
    sliders,
    html.Div(children=[
        html.Label('State Dependent Weights'),
        graph
    ], className='WeightsGraph', style={'padding': 20, 'flex': 1}),
    row_break,
    outputs
], style={'display': 'flex', 'flexDirection': 'row', 'flex-wrap': 'wrap'})


@app.callback(
    Output(component_id='d-1-star', component_property='children'),
    Input(component_id='wh-slider', component_property='value')
)
def update_d_1_star(wh):
    d_star = (1 - wh) / wh
    return rf"$d_1^*$: {round(d_star, 2)}"


@app.callback(
    Output(component_id='d-2-star', component_property='children'),
    Input(component_id='health-slider', component_property='value'),
    Input(component_id='time-slider', component_property='value')
)
def update_d_2_star(health, time):
    wh = wh_predictions[health, time]
    d_star = (1 - wh) / wh
    return rf"$d_2^*$: {round(d_star, 2)}"


@app.callback(
    Output(component_id='state-dependent-weights', component_property='figure'),
    Input(component_id='health-slider', component_property='value'),
    Input(component_id='time-slider', component_property='value')
)
def update_graph(health, time):
    wh = wh_predictions[health, time]
    fig = px.imshow(wh_predictions, color_continuous_scale='magma', zmin=0.5, zmax=1.0, origin='lower',
                    width=500, height=500, labels={'x': 'Time remaining', 'y': 'Health remaining'})
    fig.add_scatter(x=[time], y=[health], text=f"{round(wh, 2)}", marker=dict(size=15, symbol='circle'))
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])

    return fig


if __name__ == '__main__':
    app.run(debug=True)
