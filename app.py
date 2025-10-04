from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
model = joblib.load("exoplanet_model.pkl")  # load your trained model
CORS(app) 

@app.route('/')
def home():
    return "Exoplanet AI API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # get JSON input
    features = [data[f] for f in [
        'input1', 'input2', 'input3', 'input4', 'input5', 'input6',
        'input7', 'input8', 'input9', 'input10', 'input11', 'input12',
        'input13', 'input14', 'input15', 'input16', 'input17', 'input18'
    ]]
    result = model.predict([features])[0]
    return jsonify({"prediction": str(result)})

if __name__ == '__main__':
    app.run(debug=True)
