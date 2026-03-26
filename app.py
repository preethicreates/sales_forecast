from flask import Flask
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Sales Prediction Project is Running!"

@app.route('/predict')
def predict():
    # Example input (change based on your dataset)
    data = np.array([[10, 2, 2024]])
    prediction = model.predict(data)
    return f"Predicted Sales: {prediction[0]}"

app.run(host='0.0.0.0', port=10000)
