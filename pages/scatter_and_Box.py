import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn import datasets
dash.register_page(__name__,
                   path='/scatter',
                   name='Scatter and Box Plot',
                   title='Scatter and Box Plot')
# Load Iris Data
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define App Layout
layout = html.Div([
    html.H1("Iris Data Analysis"),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns[:-1]],
        value=df.columns[0]
    ),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='box-plot')
])

# Define Callbacks
@callback(
    [Output('scatter-plot', 'figure'),
     Output('box-plot', 'figure')],
    [Input('column-dropdown', 'value')]
)
def update_plots(selected_column):
    # Create Scatter Plot
    scatter_fig = px.scatter(df, x=selected_column, y='species', color='species', title=f'Scatter Plot for {selected_column}')

    # Create Box Plot
    box_fig = px.box(df, x='species', y=selected_column, color='species', title=f'Box Plot for {selected_column}')

    return scatter_fig, box_fig

# if __name__ == '__main__':
#     app.run_server(debug=True)
