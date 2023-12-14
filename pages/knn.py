# Import necessary libraries
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
dash.register_page(__name__,
                   path='/knn',
                   name='KNN classifier',
                   title='KNN')
# Initialize the Dash app
# app = dash.Dash(__name__)

# Layout of the application
layout = html.Div([
    html.H1("k-NN Classifier Performance Evaluation"),
    
    # Dropdown to select dataset
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Iris Dataset', 'value': 'iris'},
            {'label': 'Breast Cancer Dataset', 'value': 'cancer'}
        ],
        value='iris'
    ),
    
    # Dropdown to select k value
    dcc.Dropdown(
        id='k-dropdown',
        options=[
            {'label': 'k = 1', 'value': 1},
            {'label': 'k = 3', 'value': 3},
            {'label': 'k = 5', 'value': 5},
            {'label': 'k = 7', 'value': 7}
        ],
        value=1
    ),
    
    # Confusion Matrix and Metrics display
    html.Div([
        html.Div([
            html.H3("Confusion Matrix"),
            dcc.Graph(id='confusion-matrix_a')
        ], className='six columns'),
        
        html.Div([
            html.H3("Performance Metrics"),
            html.Div(id='performance-metrics_a')
        ], className='six columns'),
    ], className='row')
])

# Callback to update confusion matrix and metrics
@callback(
    [Output('confusion-matrix_a', 'figure'),
     Output('performance-metrics_a', 'children')],
    [Input('dataset-dropdown', 'value'),
     Input('k-dropdown', 'value')]
)
def update_confusion_matrix(selected_dataset, k_value):
    # Load the selected dataset
    if selected_dataset == 'iris':
        data = datasets.load_iris()
    else:
        data = datasets.load_breast_cancer()

    X = data.data
    y = data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a k-NN Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
    knn_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn_classifier.predict(X_test)

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
            'title': f'Confusion Matrix (k = {k_value})',
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
