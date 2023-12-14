# Import necessary libraries
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
dash.register_page(__name__,
                   path='/naive',
                   name='Naive Bayes',
                   title='Naive Bayes')
# Initialize the Dash app
# app = dash.Dash(__name__)

# Layout of the application
layout = html.Div([
    html.H1("Naïve Bayesian Classifier Performance Evaluation"),
    
    # Dropdown to select dataset
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Iris Dataset', 'value': 'iris'},
            {'label': 'Breast Cancer Dataset', 'value': 'cancer'}
        ],
        value='iris'
    ),
    
    # Confusion Matrix and Metrics display
    html.Div([
        html.Div([
            html.H3("Confusion Matrix"),
            dcc.Graph(id='confusion')
        ], className='six columns'),
        
        html.Div([
            html.H3("Performance Metrics"),
            html.Div(id='performance')
        ], className='six columns'),
    ], className='row')
])

# Callback to update confusion matrix and metrics
@callback(
    [Output('confusion', 'figure'),
     Output('performance', 'children')],
    [Input('dataset-dropdown', 'value')]
)
def update_confusion_matrix(selected_dataset):
    # Load the selected dataset
    if selected_dataset == 'iris':
        data = datasets.load_iris()
    else:
        data = datasets.load_breast_cancer()

    X = data.data
    y = data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a Naïve Bayesian Classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = nb_classifier.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    misclassification_rate = 1 - accuracy

    # Create confusion matrix figure
    figure = {
        'data': [{
            'z': cm,
            'type': 'heatmap',
            'colorscale': 'Viridis',
            'colorbar': {
                'title': 'Count',
                'titleside': 'right'
            }
        }],
        'layout': {
            'title': 'Confusion Matrix',
            'xaxis': {'title': 'Predicted'},
            'yaxis': {'title': 'True'}
        }
    }

    # Create performance metrics display
    metrics_display = html.Div([
        html.P(f"Recognition Rate: {accuracy * 100:.2f}%"),
        html.P(f"Misclassification Rate: {misclassification_rate * 100:.2f}%"),
        html.P(f"Sensitivity: {sensitivity:.2f}"),
        html.P(f"Specificity: {specificity:.2f}"),
        html.P(f"Precision: {precision:.2f}"),
        html.P(f"Recall: {recall:.2f}")
    ])

    return figure, metrics_display

# if __name__ == '__main__':
#     app.run_server(debug=True)
