import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import datetime
from psx import stocks, tickers


def fetch_live_data():
    tickers = tickers()
    data = stocks("SILK", start=datetime.date(
        2020, 1, 1), end=datetime.date.today())
    print("Data:\n", data)


# Load the data
df = pd.read_csv('archive/training.1600000.processed.noemoticon.csv',
                 encoding='latin-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df = df.drop(['id', 'date', 'query', 'user'], axis=1)

# Preprocess the data
df['sentiment'] = df['sentiment'].replace({0: 'negative', 4: 'positive'})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Training
# Set the experiment name
# mlflow.set_experiment('Sentiment Analysis')

# # Start a new MLflow run
# with mlflow.start_run(run_name="lr"):

#     # Vectorize the text data
#     vectorizer = TfidfVectorizer()
#     X_train_vect = vectorizer.fit_transform(X_train)

#     # Train the model
#     lr = LogisticRegression()
#     lr.fit(X_train_vect, y_train)

#     # Log the model parameters and metrics to MLflow
#     mlflow.log_param('C', lr.get_params()['C'])
#     mlflow.log_metric('train_accuracy', lr.score(X_train_vect, y_train))

#     # Vectorize the test data
#     X_test_vect = vectorizer.transform(X_test)

#     # Evaluate the model
#     y_pred = lr.predict(X_test_vect)
#     test_accuracy = accuracy_score(y_test, y_pred)

#     # Log the test accuracy to MLflow
#     mlflow.log_metric('test_accuracy', test_accuracy)

#     # Save the model to disk and log it to MLflow
#     mlflow.sklearn.log_model(lr, 'model')


# # Load all runs from experiment
# experiment_id = mlflow.get_experiment_by_name(
#     "Sentiment Analysis").experiment_id
# all_runs = mlflow.search_runs(experiment_ids=experiment_id, order_by=[
#                               "metrics.test_accuracy"])
# print(all_runs)


# Start a new MLflow run

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

# Train the model
lr = LogisticRegression()
lr.fit(X_train_vect, y_train)

# Log the model parameters and metrics to MLflow

# Vectorize the test data
X_test_vect = vectorizer.transform(X_test)

# Evaluate the model
y_pred = lr.predict(X_test_vect)
test_accuracy = accuracy_score(y_test, y_pred)

print('test_accuracy', test_accuracy)
