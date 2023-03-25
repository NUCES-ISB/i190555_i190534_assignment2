import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("stock_close_model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = tf.convert_to_tensor(np.array(float_features).reshape(1, -1))
    print('Input shape = ', features.shape)

    # prediction = model.predict([np.reshape(features, (0, 0, 3))])
    prediction = model.predict(features)
    # prediction.shape
    return render_template("index.html", prediction_text="The predicted Close value is {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)
