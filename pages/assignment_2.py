import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import pandas as pd
from scipy.stats import chi2_contingency
import base64
import io
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Initialize the Dash app
# app = dash.Dash(__name__)
dash.register_page(__name__,
                   path='/chi',
                   name='Chi Square',
                   title='Normalization')
# Define the layout for your app
layout = html.Div([
    html.H1("Correlation Analysis - Chi-Square Test"),
    # dcc.Upload(
    #     id='upload-data',
    #     children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
    #     multiple=False
    # ),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    dcc.Dropdown(
        id='attribute1-dropdown',
        # Options will be populated dynamically
    ),
    dcc.Dropdown(
        id='attribute2-dropdown',
        # Options will be populated dynamically
    ),
    html.Div(id='contingency-table-output'),
    


    html.Hr(),  # Add a horizontal line to separate normalization sections
    
        # Options will be populated dynamically
    
    html.H2("Normalization Techniques"),
    dcc.RadioItems(
        id='normalization-technique',
        options=[
            {'label': 'Min-Max Normalization', 'value': 'min-max'},
            {'label': 'Z-Score Normalization', 'value': 'z-score'},
            {'label': 'Decimal Scaling Normalization', 'value': 'decimal'}
        ],
        value='min-max',  # Default to Min-Max Normalization
    ),
    dcc.Graph(id='normalized-values-output'),
])

# Initialize an empty DataFrame
df = pd.DataFrame()

@callback(
    [Output('attribute1-dropdown', 'options'),
     Output('attribute2-dropdown', 'options')],
    [Input('upload-data', 'contents')]
)
def update_dropdowns(contents):
    global df  # Access the global df DataFrame

    # Read the CSV data into a DataFrame
    if contents is None:
        raise dash.exceptions.PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Create dropdown options from column names
    attribute_options = [{'label': col, 'value': col} for col in df.columns]

    return attribute_options, attribute_options

@callback(
    Output('contingency-table-output', 'children'),
    [Input('attribute1-dropdown', 'value'),
     Input('attribute2-dropdown', 'value')]
)
def chi_square_test(attr1, attr2):
    if not attr1 or not attr2:
        raise dash.exceptions.PreventUpdate

    # Create the contingency table
    contingency_table = pd.crosstab(df[attr1], df[attr2])

    # Perform the Chi-Square Test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Determine if the attributes are correlated
    if p < 0.05:
        conclusion = 'Attributes are correlated.'
    else:
        conclusion = 'Attributes are not correlated.'

    return html.Div([
        html.H3('Contingency Table'),
        dcc.Graph(figure=px.imshow(contingency_table)),
        html.H3('Chi-Square Value: {:.2f}'.format(chi2)),
        html.H3('Conclusion: {}'.format(conclusion))
    ])


@callback(
    Output('normalized-values-output', 'figure'),
    [Input('attribute1-dropdown', 'value'),
     Input('attribute2-dropdown', 'value'),
     Input('normalization-technique', 'value')]
)
def normalize_selected_attributes(attr1, attr2, normalization_technique):
    if not attr1 or not attr2:
        raise dash.exceptions.PreventUpdate

    selected_attributes = [attr1, attr2]
    selected_df = df[selected_attributes]

    if normalization_technique == 'min-max':
        scaler = MinMaxScaler()
    elif normalization_technique == 'z-score':
        scaler = StandardScaler()
    elif normalization_technique == 'decimal':
        scaler = StandardScaler(with_mean=False)

    normalized_data = scaler.fit_transform(selected_df)

    # Create a scatter plot of the normalized data
    figure = {
        'data': [
            {
                'x': normalized_data[:, 0],
                'y': normalized_data[:, 1],
                'mode': 'markers',
                'type': 'scatter',
                'text': df['Species'],
                'marker': {'size': 10, 'opacity': 0.5}
            }
        ],
        'layout': {
            'xaxis': {'title': attr1},
            'yaxis': {'title': attr2},
            'title': f'Normalized Data ({normalization_technique})'
        }
    }

    return figure


