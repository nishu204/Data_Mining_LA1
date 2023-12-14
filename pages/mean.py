import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from scipy import stats
import base64
import io

# app = dash.Dash(__name__)
dash.register_page(__name__,
                   path='/datades',
                   name='Data Description',
                   title='Data Description')

layout = html.Div([
    html.H1("Data Analysis Dashboard"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
        },
        multiple=False
    ),
    html.Div(id='output-data-upload_b'),
])

def parse_contents(contents):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Drop the last column (assuming it's the label column)
    df = df.iloc[:, :-1]

    return df

@callback(Output('output-data-upload_b', 'children'),
              Input('upload-data', 'contents'))
def update_output(contents):
    if contents is None:
        return html.Div(['Upload a CSV file to get started.'])

    df = parse_contents(contents)

    central_tendency = {
        'Mean': df.mean().to_dict(),
        'Median': df.median().to_dict(),
        'Mode': df.mode().iloc[0].to_dict(),
        'Midrange': ((df.min() + df.max()) / 2).to_dict(),
        'Variance': df.var().to_dict(),
        'Standard Deviation': df.std().to_dict()
    }

    dispersion = {
        'Range': (df.max() - df.min()).to_dict(),
        'Quartiles': df.quantile([0.25, 0.75]).to_dict(),
        'Interquartile Range (IQR)': stats.iqr(df).item(),
        'Five-Number Summary': [df.min().to_dict(), df.quantile(0.25).to_dict(), 
                               df.median().to_dict(), df.quantile(0.75).to_dict(), 
                               df.max().to_dict()]
    }

    central_tendency_html = html.Div([
        html.H2('Measures of Central Tendency'),
        html.Table(
            # Header
            [html.Tr([html.Th('Measure'), html.Th('Value')])] +

            # Rows
            [html.Tr([html.Td(k), html.Td(str(v))]) for k, v in central_tendency.items()]
        )
    ])

    dispersion_html = html.Div([
        html.H2('Measures of Dispersion'),
        html.Table(
            # Header
            [html.Tr([html.Th('Measure'), html.Th('Value')])] +

            # Rows
            [html.Tr([html.Td(k), html.Td(str(v))]) for k, v in dispersion.items()]
        )
    ])

    return [
        central_tendency_html,
        html.Br(),
        dispersion_html
    ]

# if __name__ == '__main__':
#     app.run_server(debug=True)
