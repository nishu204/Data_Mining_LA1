import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import datasets
import scipy.stats as stats
dash.register_page(__name__)
# Load Iris Data
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Initialize Dash App
# app = dash.Dash(__name__)

# Define App Layout
layout = html.Div([
    html.H1("Iris Data Analysis"),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=df.columns[0]
    ),
    dcc.Dropdown(
        id='plot-type-dropdown',
        options=[
            {'label': 'Quantile Plot', 'value': 'quantile'},
            {'label': 'Quantile-Quantile Plot', 'value': 'qq-plot'},
            {'label': 'Histogram', 'value': 'histogram'},
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Boxplot', 'value': 'boxplot'}
        ],
        value='quantile'
    ),
    dcc.Graph(id='selected-plot'),
    html.Div(id='stats-output')
])

# Define Callback Functions
@callback(
    [Output('selected-plot', 'figure'),
     Output('stats-output', 'children')],
    [Input('column-dropdown', 'value'),
     Input('plot-type-dropdown', 'value')]
)
def update_plot(selected_column, selected_plot_type):
    # Calculate statistics
    mean = df[selected_column].mean()
    median = df[selected_column].median()
    mode = df[selected_column].mode().iloc[0]
    midrange = (max(df[selected_column]) + min(df[selected_column])) / 2
    variance = df[selected_column].var()
    std_dev = df[selected_column].std()
    data_range = max(df[selected_column]) - min(df[selected_column])
    quartiles = df[selected_column].quantile([0.25, 0.5, 0.75])
    iqr = quartiles[0.75] - quartiles[0.25]
    five_num_summary = [min(df[selected_column]), quartiles[0.25], median, quartiles[0.75], max(df[selected_column])]

    stats_html = f"Selected Column: {selected_column}"
    stats_html += f"<b>Mean:</b> {mean}<br>"
    stats_html += f"<b>Median:</b> {median}<br>"
    stats_html += f"<b>Mode:</b> {mode}<br>"
    stats_html += f"<b>Midrange:</b> {midrange}<br>"
    stats_html += f"<b>Variance:</b> {variance}<br>"
    stats_html += f"<b>Standard Deviation:</b> {std_dev}<br>"
    stats_html += f"<b>Range:</b> {data_range}<br>"
    stats_html += f"<b>Quartiles:</b> Q1: {quartiles[0.25]}, Median: {median}, Q3: {quartiles[0.75]}<br>"
    stats_html += f"<b>Interquartile Range:</b> {iqr}<br>"
    stats_html += f"<b>Five-Number Summary:</b> Min: {five_num_summary[0]}, Q1: {five_num_summary[1]}, Median: {five_num_summary[2]}, Q3: {five_num_summary[3]}, Max: {five_num_summary[4]}"

    # Create the selected plot based on plot type
    if selected_plot_type == 'quantile':
        fig = px.scatter(x=np.random.normal(size=len(df[selected_column])),
                         y=np.sort(df[selected_column]),
                         labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
        x = np.linspace(np.min(df[selected_column]), np.max(df[selected_column]), len(df[selected_column]))
        y = np.quantile(df[selected_column], np.linspace(0, 1, len(df[selected_column])))
        fig.update_layout(title=f'Quantile Plot for {selected_column}',
                          xaxis_title='Theoretical Quantiles',
                          yaxis_title='Sample Quantiles')
        fig.add_shape(type='line',
                      x0=min(x),
                      y0=min(y),
                      x1=max(x),
                      y1=max(y),
                      line=dict(color='red', width=2, dash='dash'))
    elif selected_plot_type == 'qq-plot':
        fig = px.scatter(x=np.random.normal(size=len(df[selected_column])),
                         y=np.sort(df[selected_column]),
                         labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
        x = np.linspace(np.min(df[selected_column]), np.max(df[selected_column]), len(df[selected_column]))
        y = np.quantile(df[selected_column], np.linspace(0, 1, len(df[selected_column])))
        fig.update_layout(title=f'Q-Q Plot for {selected_column}',
                          xaxis_title='Theoretical Quantiles',
                          yaxis_title='Sample Quantiles')
        fig.add_shape(type='line',
                      x0=min(x),
                      y0=min(y),
                      x1=max(x),
                      y1=max(y),
                      line=dict(color='red', width=2, dash='dash'))
    elif selected_plot_type == 'histogram':
        fig = px.histogram(df, x=selected_column, nbins=30, labels={selected_column: 'Value'})
        fig.update_layout(title=f'Histogram for {selected_column}')
    elif selected_plot_type == 'scatter':
        fig = px.scatter(df, x=selected_column, y='species', color='species', labels={selected_column: 'Value'})
        fig.update_layout(title=f'Scatter Plot for {selected_column}')

    elif selected_plot_type == 'boxplot':
        fig = px.box(df, x='species', y=selected_column, color='species', labels={selected_column: 'Value'})
        fig.update_layout(title=f'Boxplot for {selected_column}')

    return fig, stats_html

# if __name__ == '__main__':
#     app.run_server(debug=True)
