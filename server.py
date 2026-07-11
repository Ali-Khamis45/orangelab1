import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')

# Load the serialized model, scaler, and columns
with open('ml_models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('ml_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ml_models/feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

with open('ml_models/defaults.json', 'r') as f:
    defaults = json.load(f)

# Serve the static front-end page
@app.route('/')
def index():
    return send_from_directory(app.static_folder or 'static', 'index.html')

# Expose prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Initialize a single row representation with 0s for all columns
        row_dict = {col: 0.0 for col in feature_columns}
        
        # 1. Populate numerical columns
        num_cols = ['Age', 'Tenure', 'Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score', 'Training Duration(Days)', 'Training Cost']
        for col in num_cols:
            val = data.get(col)
            if val is not None:
                row_dict[col] = float(val)
            else:
                row_dict[col] = float(defaults.get(col, 0.0))
        
        # 2. Populate categorical columns (creating dummy variable names)
        cat_cols = ['GenderCode', 'MaritalDesc', 'RaceDesc', 'DepartmentType', 'BusinessUnit', 'EmployeeType', 'PayZone', 'EmployeeClassificationType', 'Training Type', 'Training Program Name']
        for col in cat_cols:
            val = data.get(col)
            if val is not None:
                dummy_name = f"{col}_{val}"
                if dummy_name in row_dict:
                    row_dict[dummy_name] = 1.0
        
        # Convert dictionary to DataFrame with columns in exact order
        X_df = pd.DataFrame([row_dict], columns=feature_columns)
        
        # Scale features
        X_scaled = scaler.transform(X_df)
        
        # Predict probability and class label
        prob = float(model.predict_proba(X_scaled)[0][1])
        prediction = int(model.predict(X_scaled)[0])
        
        # Calculate risk contributors based on model coefficients
        # coefficient * scaled_value shows contribution size/direction
        coefficients = model.coef_[0]
        contributions = {}
        for idx, col in enumerate(feature_columns):
            contrib = float(coefficients[idx] * X_scaled[0][idx])
            contributions[col] = contrib
            
        # Group contributions into logical display categories
        display_contribs = {
            "Tenure Impact": contributions.get("Tenure", 0.0),
            "Age Impact": contributions.get("Age", 0.0),
            "Satisfaction & Balance": contributions.get("Satisfaction Score", 0.0) + contributions.get("Work-Life Balance Score", 0.0),
            "Engagement & Morale": contributions.get("Engagement Score", 0.0),
            "Training Investment": contributions.get("Training Cost", 0.0) + contributions.get("Training Duration(Days)", 0.0)
        }
        
        # Return response
        return jsonify({
            'status': 'success',
            'churn_probability': prob,
            'prediction': prediction,
            'contributions': display_contribs,
            'risk_level': 'High Risk' if prob >= 0.5 else 'Low Risk'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Create static directory if not exists
    os.makedirs('static', exist_ok=True)
    app.run(host='127.0.0.1', port=5000, debug=True)
