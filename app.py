# app.py
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.secret_key = 'replace_with_a_strong_secret'  # change before production

# Paths
DATA_PATH = 'data/final_commercial_crops_karnataka.csv'
MODEL_PATH = 'model/crop_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

# Load dataset and model
df = pd.read_csv(DATA_PATH, encoding='latin-1')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Dummy users (replace with DB later)
users = {'admin': 'admin'}

@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        if users.get(u) == p:
            session['username'] = u
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'username' not in session:
        return redirect(url_for('login'))

    family = request.form.get('family', '')
    fertilizer = request.form.get('fertilizer', '')
    try:
        nitrogen = float(request.form.get('nitrogen', 0))
        phosphorus = float(request.form.get('phosphorus', 0))
        potassium = float(request.form.get('potassium', 0))
    except ValueError:
        return render_template('index.html', username=session['username'], error="Enter numeric values for N,P,K")

    # Prepare input similarly to training
    input_df = pd.DataFrame({
        'Family': [family],
        'Fertilizer Used': [fertilizer],
        'Nitrogen (N) (%)': [nitrogen],
        'Phosphorus (P) (%)': [phosphorus],
        'Potassium (K) (%)': [potassium]
    })
    X_train = pd.get_dummies(df[['Family', 'Fertilizer Used', 'Nitrogen (N) (%)',
                                'Phosphorus (P) (%)', 'Potassium (K) (%)']], drop_first=True)
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    # scale
    num_cols = ['Nitrogen (N) (%)', 'Phosphorus (P) (%)', 'Potassium (K) (%)']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)[0]

    return render_template('result.html', username=session['username'], result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
