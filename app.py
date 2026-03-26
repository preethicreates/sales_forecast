import numpy as np
import pickle
from flask import Flask

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Sales Prediction Project is Running!"

@app.route('/predict')
def predict():
    try:
        # IMPORTANT: change 20 → your actual number of features
        data = np.zeros((1, 30))  
        
        prediction = model.predict(data)
        return f"Prediction: {prediction[0]}"
    
    except Exception as e:
        return f"Error: {str(e)}"

app.run(host='0.0.0.0', port=10000)
