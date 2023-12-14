import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from scipy.stats import probplot  # Import for Q-Q plot

from sklearn import datasets

dash.register_page(__name__,
                   path='/qq-plot-histogram',
                   name='Q-Q Plot and Histogram',
                   title='Q-Q Plot and Histogram')

# Load Iris Data
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Initialize Dash App
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define App Layout
layout = html.Div([
    html.H1("Iris Data Q-Q Plot and Histogram"),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns[:-1]],
        value=df.columns[0]
    ),
    dcc.Graph(id='qq-plot_a'),
    dcc.Graph(id='histogram_a')
])

# Define Callbacks
@callback(
    [Output('qq-plot_a', 'figure'),
     Output('histogram_a', 'figure')],
    [Input('column-dropdown', 'value')]
)
def update_plots(selected_column):
    # Create Q-Q Plot
    _, ax = probplot(df[selected_column], plot=None)
    qq_plot_df = pd.DataFrame({'Theoretical Quantiles': ax[0], 'Ordered Values': ax[1]})
    qq_plot_fig = px.scatter(qq_plot_df, x='Theoretical Quantiles', y='Ordered Values',
                            title=f'Q-Q Plot for {selected_column}')

    # Create Histogram
    histogram_fig = px.histogram(df, x=selected_column, title=f'Histogram for {selected_column}')

    return qq_plot_fig, histogram_fig

# if __name__ == '__main__':
#     app.run_server(debug=True)
