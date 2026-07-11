import os
import json
import pickle
import datetime
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
        
        # Determine confidence score (probability of the predicted class)
        confidence = float(prob if prediction == 1 else (1.0 - prob))
        
        # Assign risk category
        if prob >= 0.70:
            risk_cat = "Critical Risk"
        elif prob >= 0.50:
            risk_cat = "High Risk"
        elif prob >= 0.30:
            risk_cat = "Moderate Risk"
        else:
            risk_cat = "Low Risk"
            
        # Generate dynamic explanation sentence
        sorted_contribs = sorted(display_contribs.items(), key=lambda x: x[1], reverse=True)
        top_pos = next((name for name, val in sorted_contribs if val > 0.02), None)
        top_neg = next((name for name, val in reversed(sorted_contribs) if val < -0.02), None)
        
        explanation_parts = []
        if prediction == 1:
            explanation_parts.append(f"Attrition predicted with {confidence*100:.1f}% confidence.")
            if top_pos:
                explanation_parts.append(f"The primary driver increasing risk is {top_pos}.")
            if top_neg:
                explanation_parts.append(f"This is partially mitigated by protective factors in {top_neg}.")
        else:
            explanation_parts.append(f"Retention predicted with {confidence*100:.1f}% confidence.")
            if top_neg:
                explanation_parts.append(f"The primary protective factor is {top_neg}.")
            if top_pos:
                explanation_parts.append(f"Minor risk is introduced by {top_pos}.")
                
        explanation = " ".join(explanation_parts) if explanation_parts else "Prediction is stable with baseline metrics."
        
        # Generate timestamp and model version info
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_ver = "Tuned Gradient Boosting Classifier v2.0"
        
        # Return response
        return jsonify({
            'status': 'success',
            'churn_probability': prob,
            'prediction': prediction,
            'contributions': display_contribs,
            'risk_level': 'High Risk' if prob >= 0.5 else 'Low Risk',
            'confidence_score': confidence,
            'risk_category': risk_cat,
            'explanation': explanation,
            'prediction_timestamp': timestamp,
            'model_version': model_ver
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
