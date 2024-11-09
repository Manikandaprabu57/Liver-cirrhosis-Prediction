from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route to check if the server is running
@app.route('/', methods=['GET'])
def home():
    return "Server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['Age'], data['Bilirubin'], data['Albumin'], data['Ascites'], data['Sex']]])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Make a prediction
    prediction = model.predict(features_scaled)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
