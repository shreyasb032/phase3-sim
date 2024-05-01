import pickle
import plotly.express as px
from dash import dcc

with open('../models/weights_for_plot.pkl', 'rb') as f:
    wh_predictions = pickle.load(f)

fig = px.imshow(wh_predictions, color_continuous_scale='magma', zmin=0.5, zmax=1.0, origin='lower',
                width=500, height=500, labels={'x': 'Time remaining', 'y': 'Health remaining'})
graph = dcc.Graph(id='state-dependent-weights',
                  figure=fig)
