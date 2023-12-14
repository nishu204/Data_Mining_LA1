import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import pearsonr
import numpy as np
dash.register_page(__name__,
                   path='/corr',
                   name='Pearson Coefficient',
                   title='Pearson Coefficient')
# Load the Iris dataset
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# app = dash.Dash(__name__)

layout = html.Div([
    html.H1("Correlation Analysis with Iris Data"),
    dcc.Dropdown(
        id='attribute1-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=df.columns[0]
    ),
    dcc.Dropdown(
        id='attribute2-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=df.columns[1]
    ),
    html.Div(id='correlation-output'),
])

@callback(
    Output('correlation-output', 'children'),
    Input('attribute1-dropdown', 'value'),
    Input('attribute2-dropdown', 'value')
)
def update_correlation(attribute1, attribute2):
    selected_data = df[[attribute1, attribute2]]
    correlation_coefficient, _ = pearsonr(selected_data[attribute1], selected_data[attribute2])
    cov = np.cov(selected_data[attribute1], selected_data[attribute2])[0][1]

    if correlation_coefficient > 0:
        conclusion = f"The selected attributes '{attribute1}' and '{attribute2}' are positively correlated."
    elif correlation_coefficient < 0:
        conclusion = f"The selected attributes '{attribute1}' and '{attribute2}' are negatively correlated."
    else:
        conclusion = f"There is no linear correlation between '{attribute1}' and '{attribute2}'."

    return html.Div([
        html.H3(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}"),
        html.H3(f"Covariance: {cov:.2f}"),
        html.P(conclusion)
    ])

# if __name__ == '__main__':
#     app.run_server(debug=True)
