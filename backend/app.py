from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Ensure the model loads correctly relative to this file
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "exoplanet_model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data.get(f, 0) for f in [
        "koi_score","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co",
        "koi_fpflag_ec","koi_period","koi_time0bk","koi_impact",
        "koi_duration","koi_depth","koi_prad","koi_teq","koi_insol",
        "koi_model_snr","koi_steff","koi_slogg","koi_srad","koi_kepmag"
    ]]
    result = model.predict([features])[0]
    return jsonify({"prediction": str(result)})

if __name__ == '__main__':
    app.run(debug=True)
