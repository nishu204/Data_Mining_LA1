import dash
from dash import dcc, html, callback
from dash import dash_table
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dash.dash_table.Format import Group
dash.register_page(__name__,
                   path='/rule',
                   name='Rule Based Classifier',
                   title='Rule Based')
# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rule-based classifier function (you can customize this)
def rule_based_classifier(data):
    rules = []
    
    # Define your rules here

    # Example rule: If mean radius is less than 15, classify as malignant, else benign
    for index, row in data.iterrows():
        if row["mean radius"] < 15:
            rules.append(1)  # Malignant
        else:
            rules.append(0)  # Benign
 # Benign

    # Example Rule 2: If mean texture is greater than 20, classify as malignant (1), else benign (0)
    # for index, row in data.iterrows():
    #     if row["mean texture"] > 20:
    #         rules[index] = 1  # Malignant
    #     else:
    #         rules[index] = 0  # Benign
    
    return rules

# Calculate coverage, accuracy, and toughness (size)
def evaluate_performance(y_true, y_pred):
    coverage = sum(y_pred) / len(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    toughness = len(y_pred)
    return coverage, accuracy, toughness

# Create Dash app
# app = dash.Dash(__name__)

layout = html.Div([
    html.H1("Rule-Based Classifier Performance Evaluation"),
    
    dcc.Graph(id="coverage-accuracy-plot"),
    
    html.Div(id="performance-metrics"),
    
    dash_table.DataTable(
        id="rule-results",
        columns=[{"name": "Sample Index", "id": "index"}, {"name": "Predicted Class", "id": "predicted_class"}],
        style_table={'height': '300px', 'overflowY': 'auto'},
    )
])

@callback(
    [Output("performance-metrics", "children"),
     Output("rule-results", "data"),
     Output("coverage-accuracy-plot", "figure")],
    [Input("rule-results", "selected_rows")]
)
def update_results(selected_rows):
    # Apply the rule-based classifier to the test data
    y_pred = rule_based_classifier(X_test)
    
    # Evaluate performance
    coverage, accuracy, toughness = evaluate_performance(y_test, y_pred)
    
    # Create a DataFrame to display results in the Dash table
    results_df = pd.DataFrame({"index": X_test.index, "predicted_class": y_pred})
    
    # Create a coverage vs. accuracy plot
    coverage_accuracy_plot = {
        "data": [
            {"x": [coverage], "y": [accuracy], "type": "scatter", "mode": "markers+text", "text": ["Rule-Based Classifier"]}
        ],
        "layout": {
            "xaxis": {"title": "Coverage"},
            "yaxis": {"title": "Accuracy"},
            "title": "Coverage vs. Accuracy",
        },
    }
    
    # Display performance metrics
    performance_metrics = html.Div([
        html.H3(f"Coverage: {coverage:.2%}"),
        html.H3(f"Accuracy: {accuracy:.2%}"),
        html.H3(f"Toughness (Size): {toughness}")
    ])
    
    return performance_metrics, results_df.to_dict("records"), coverage_accuracy_plot

# if __name__ == '__main__':
#     app.run_server(debug=True)
