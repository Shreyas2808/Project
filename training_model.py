# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs('model', exist_ok=True)

# 1. Load data
df = pd.read_csv('data/final_commercial_crops_karnataka.csv', encoding='latin-1')

# 2. Features and label
X = pd.get_dummies(df[['Family', 'Fertilizer Used', 'Nitrogen (N) (%)',
                       'Phosphorus (P) (%)', 'Potassium (K) (%)']], drop_first=True)
y = df['Crop Name']

# 3. Fit scaler on numeric columns (make sure columns exist)
num_cols = ['Nitrogen (N) (%)', 'Phosphorus (P) (%)', 'Potassium (K) (%)']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 4. Train model
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X, y)

# 5. Save model and scaler
joblib.dump(model, 'model/crop_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Model and scaler saved to model/ directory")
