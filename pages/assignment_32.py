import dash
from dash import dcc, html, get_asset_url, callback
from dash.dependencies import Input, Output
import os
dash.register_page(__name__,
                   path='/vis',
                   name='Decision Tree(Visual)',
                   title='Visual')


# Define the app layout
layout = html.Div([
    html.H1("Decision Tree Classifier Images"),
    dcc.Dropdown(
        id='image-selector',
        options=[
            {'label': 'Information Gain', 'value': 'info.png'},
            {'label': 'Gini Index', 'value': 'gini.png'},
            {'label': 'Gain Ratio', 'value': 'gain.png'}
        ],
        value='info.png',
        clearable=False
    ),
    html.Img(id='selected-image', style={'width': '80%'})
])

# Define a callback to update the displayed image based on dropdown selection
@callback(
    Output('selected-image', 'src'),
    [Input('image-selector', 'value')]
)
def update_image(selected_value):
    image_path = get_asset_url(selected_value)
    return image_path
