from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'replace_with_a_strong_secret'

# ---------- Database Setup ----------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- User Model ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Create tables (run once)
with app.app_context():
    db.create_all()

# ---------- Load Dataset ----------
DATA_PATH = 'data/final_commercial_crops_karnataka.csv'
df = pd.read_csv(DATA_PATH, encoding='latin-1')

# ---------- Routes ----------
@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

# ---------------- SIGN UP ----------------
@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['new_username']
    password = request.form['new_password']
    confirm = request.form['confirm_password']

    if password != confirm:
        return render_template('login.html', signup_error='Passwords do not match')

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return render_template('login.html', signup_error='Username already exists')

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()
    return render_template('login.html', error='Account created! Please log in.')

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ---------------- RECOMMEND ----------------
@app.route('/recommend', methods=['POST'])
def recommend():
    if 'username' not in session:
        return redirect(url_for('login'))

    previous_crop = request.form.get('previous_crop', '').strip().title()

    # Find fertilizer used for this crop
    match = df[df['Crop'].str.title() == previous_crop]

    if match.empty:
        return render_template(
            'index.html',
            username=session['username'],
            error=f"No data found for {previous_crop}."
        )

    fertilizer_used = match['Fertilizer Used'].iloc[0]

    # Recommend other crops using the same fertilizer
    recommended_crops = df[df['Fertilizer Used'] == fertilizer_used]['Crop'].unique().tolist()

    # Remove the previous crop itself from recommendations
    recommended_crops = [crop for crop in recommended_crops if crop.title() != previous_crop]

    return render_template(
        'result.html',
        username=session['username'],
        previous_crop=previous_crop,
        fertilizer=fertilizer_used,
        crops=recommended_crops
    )

if __name__ == '__main__':
    app.run(debug=True)
