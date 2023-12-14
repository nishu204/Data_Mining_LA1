import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
dash.register_page(__name__,
                   path='/dtree',
                   name='Decision Tree',
                   title='Decision Tree')
# Load breast cancer dataset
data = datasets.load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Dash app
# app = dash.Dash(__name__)

# Define the app layout
layout = html.Div([
    html.H1("Decision Tree Classifier with Attribute Selection Measures"),
    dcc.Dropdown(
        id='attribute-selector',
        options=[
            {'label': 'Information Gain', 'value': 'information_gain'},
            {'label': 'Gain Ratio', 'value': 'gain_ratio'},
            {'label': 'Gini Index', 'value': 'gini_index'}
        ],
        value='information_gain',
        clearable=False
    ),
    dcc.Graph(id='confusion-matrix'),
    html.Div(id='evaluation-metrics')
])

# Define a function to train and evaluate the classifier
def train_and_evaluate(attribute_selector):
    # Train the Decision Tree classifier with the selected attribute selection measure
    if attribute_selector == 'information_gain':
        criterion = 'entropy'
    elif attribute_selector == 'gain_ratio':
        criterion = 'gini'
    else:
        criterion = 'gini'
    
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return cm, accuracy, precision, recall

# Define a callback function to update the graph and evaluation metrics
@callback(
    [Output('confusion-matrix', 'figure'),
     Output('evaluation-metrics', 'children')],
    [Input('attribute-selector', 'value')]
)
def update_metrics(attribute_selector):
    cm, accuracy, precision, recall = train_and_evaluate(attribute_selector)
    
    # Create the confusion matrix figure
    cm_fig = {
        'data': [{
            'type': 'heatmap',
            'z': cm,
            'x': ['Predicted Benign', 'Predicted Malignant'],
            'y': ['Benign', 'Malignant'],
            'colorscale': 'Viridis'
        }],
        'layout': {
            'title': 'Confusion Matrix',
            'xaxis': {'title': 'Predicted Labels'},
            'yaxis': {'title': 'Actual Labels'}
        }
    }
    
    # Create the evaluation metrics text
    metrics_text = html.Div([
        html.P(f'Accuracy: {accuracy:.2f}'),
        
        html.P(f'Precision: {precision:.2f}'),
        html.P(f'Recall: {recall:.2f}')
    ])
    
    return cm_fig, metrics_text

# Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)
