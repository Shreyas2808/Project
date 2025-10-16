import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('data/final_commercial_crops_karnataka.csv', encoding='latin-1')

# Use top 10 crops only
top_crops = df['Crop Name'].value_counts().nlargest(10).index
df_filtered = df[df['Crop Name'].isin(top_crops)]

# Prepare features and target
X = pd.get_dummies(df_filtered[['Family', 'Fertilizer Used', 'Nitrogen (N) (%)',
                                'Phosphorus (P) (%)', 'Potassium (K) (%)']], drop_first=True)
y = df_filtered['Crop Name']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric columns
scaler = StandardScaler()
num_cols = ['Nitrogen (N) (%)', 'Phosphorus (P) (%)', 'Potassium (K) (%)']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Compare all models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results.append((name, acc * 100, f1 * 100))

# Print results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "F1 Score (%)"])
print(results_df)