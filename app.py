"""
app.py - Smart Lender Flask Application
Main web application for real-time loan prediction powered by XGBoost.
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

app = Flask(__name__)
app.secret_key = 'smartlender_secret_2024'

# ── Load model pipeline ────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join('models', 'smart_lender_xgb.joblib')
METRICS_PATH = os.path.join('data', 'model_metrics.json')

pipeline = None
metrics  = None

def load_assets():
    global pipeline, metrics
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print("[INFO] Model pipeline loaded.")
    else:
        print("[WARNING] Model not found. Please run scripts/train_models.py first.")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        print("[INFO] Metrics loaded.")

load_assets()


def preprocess_input(form_data: dict) -> np.ndarray:
    """Preprocess a single applicant's form data using the saved pipeline."""
    cat_cols = pipeline['categorical_cols']
    num_cols = pipeline['numerical_cols']
    encoders = pipeline['encoders']
    scaler   = pipeline['scaler']
    cat_imp  = pipeline['cat_imputer']
    num_imp  = pipeline['num_imputer']

    row = {
        'Gender':            form_data.get('gender', 'Male'),
        'Married':           form_data.get('married', 'No'),
        'Dependents':        form_data.get('dependents', '0'),
        'Education':         form_data.get('education', 'Graduate'),
        'Self_Employed':     form_data.get('self_employed', 'No'),
        'ApplicantIncome':   float(form_data.get('applicant_income', 5000)),
        'CoapplicantIncome': float(form_data.get('coapplicant_income', 0)),
        'LoanAmount':        float(form_data.get('loan_amount', 150)),
        'Loan_Amount_Term':  float(form_data.get('loan_term', 360)),
        'Credit_History':    float(form_data.get('credit_history', 1.0)),
        'Property_Area':     form_data.get('property_area', 'Urban'),
    }

    df = pd.DataFrame([row])

    # Impute
    df[cat_cols] = cat_imp.transform(df[cat_cols])
    df[num_cols] = num_imp.transform(df[num_cols])

    # Encode
    for col in cat_cols:
        le = encoders[col]
        val = df[col].astype(str).values[0]
        if val not in le.classes_:
            val = le.classes_[0]
        df[col] = le.transform([val])

    # Scale
    df[num_cols] = scaler.transform(df[num_cols])

    return df[pipeline['feature_names']].values


def preprocess_batch(df_raw: pd.DataFrame):
    """Preprocess a batch DataFrame for prediction."""
    cat_cols = pipeline['categorical_cols']
    num_cols = pipeline['numerical_cols']
    encoders = pipeline['encoders']
    scaler   = pipeline['scaler']
    cat_imp  = pipeline['cat_imputer']
    num_imp  = pipeline['num_imputer']

    df = df_raw.copy()
    # Drop Loan_ID if present
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    if 'Loan_Status' in df.columns:
        df = df.drop(columns=['Loan_Status'])

    df[cat_cols] = cat_imp.transform(df[cat_cols])
    df[num_cols] = num_imp.transform(df[num_cols].apply(pd.to_numeric, errors='coerce'))

    for col in cat_cols:
        le = encoders[col]
        df[col] = df[col].astype(str).apply(
            lambda v: v if v in le.classes_ else le.classes_[0]
        )
        df[col] = le.transform(df[col])

    df[num_cols] = scaler.transform(df[num_cols])
    return df[pipeline['feature_names']].values


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    ctx = {'metrics': metrics, 'model_loaded': pipeline is not None}
    return render_template('index.html', **ctx)


@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html', metrics=metrics)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503

    data = request.get_json() if request.is_json else request.form.to_dict()

    try:
        X = preprocess_input(data)
        model = pipeline['model']
        pred  = model.predict(X)[0]
        prob  = model.predict_proba(X)[0]

        approved     = bool(pred == 1)
        confidence   = float(max(prob)) * 100
        approve_prob = float(prob[1]) * 100
        reject_prob  = float(prob[0]) * 100

        # Risk level classification
        if approve_prob >= 75:
            risk_level = 'Low Risk'
            risk_color = 'green'
        elif approve_prob >= 50:
            risk_level = 'Moderate Risk'
            risk_color = 'amber'
        else:
            risk_level = 'High Risk'
            risk_color = 'red'

        return jsonify({
            'approved':     approved,
            'prediction':   'Approved' if approved else 'Rejected',
            'confidence':   round(confidence, 2),
            'approve_prob': round(approve_prob, 2),
            'reject_prob':  round(reject_prob, 2),
            'risk_level':   risk_level,
            'risk_color':   risk_color,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/batch', methods=['GET'])
def batch_page():
    return render_template('batch.html', metrics=metrics)


@app.route('/api/batch', methods=['POST'])
def api_batch():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        df_raw = pd.read_csv(file)
        loan_ids = df_raw['Loan_ID'].tolist() if 'Loan_ID' in df_raw.columns else [f'APP-{i+1:04d}' for i in range(len(df_raw))]

        X_batch = preprocess_batch(df_raw)
        model   = pipeline['model']
        preds   = model.predict(X_batch)
        probs   = model.predict_proba(X_batch)

        results = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            approve_prob = float(prob[1]) * 100
            if approve_prob >= 75:
                risk = 'Low Risk'
            elif approve_prob >= 50:
                risk = 'Moderate Risk'
            else:
                risk = 'High Risk'

            results.append({
                'loan_id':      loan_ids[i],
                'prediction':   'Approved' if pred == 1 else 'Rejected',
                'approved':     bool(pred == 1),
                'approve_prob': round(approve_prob, 2),
                'reject_prob':  round(float(prob[0]) * 100, 2),
                'risk_level':   risk,
            })

        summary = {
            'total':    len(results),
            'approved': sum(1 for r in results if r['approved']),
            'rejected': sum(1 for r in results if not r['approved']),
        }
        summary['approval_rate'] = round(summary['approved'] / summary['total'] * 100, 1)

        return jsonify({'results': results, 'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/batch/download', methods=['POST'])
def download_batch():
    data = request.get_json()
    results = data.get('results', [])
    df_out = pd.DataFrame(results)
    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='smart_lender_predictions.csv'
    )


@app.route('/api/metrics')
def api_metrics():
    return jsonify(metrics if metrics else {})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
