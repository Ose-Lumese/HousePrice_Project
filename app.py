from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model from the /model/ folder as per brief
model_path = os.path.join('model', 'house_price_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract the 6 features in the correct order
        input_data = np.array([[
            int(data['OverallQual']),
            int(data['GrLivArea']),
            int(data['GarageCars']),
            int(data['FullBath']),
            int(data['YearBuilt']),
            int(data['TotalBsmtSF'])
        ]])
        
        prediction = model.predict(input_data)[0]
        return jsonify({'price': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)