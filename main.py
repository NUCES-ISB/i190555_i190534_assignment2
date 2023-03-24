from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
with open('stock_close_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    n_steps = 2
    n_features = 3
    # Get the input from the user
    input_data = request.form.to_dict()
    input_data = {k: float(v) for k, v in input_data.items()}
    print(input_data)
    # Convert the input into a DataFrame
    input_df = pd.DataFrame.from_dict([input_data])

    # Reshape the input data to have a batch size of 1
    input_data_reshaped = np.reshape(input_df.values, (1, n_steps, n_features))

    # Make a prediction
    prediction = model.predict(input_data_reshaped)

    # Format the prediction as a string
    prediction_str = f'${prediction[0][0]:,.2f}'

    # Render the prediction page with the result
    return render_template('predict.html', prediction=prediction_str)

    # Get the input from the user
    input_data = request.form.to_dict()
    
    input_data = {k: float(v) for k, v in input_data.items()}

    # Convert the input into a DataFrame
    input_df = pd.DataFrame.from_dict([input_data])

    # Make a prediction
    prediction = model.predict(input_df)

    # Format the prediction as a string
    prediction_str = f'${prediction[0][0]:,.2f}'

    # Render the prediction page with the result
    return render_template('predict.html', prediction=prediction_str)


if __name__ == '__main__':
    app.run(debug=True)
