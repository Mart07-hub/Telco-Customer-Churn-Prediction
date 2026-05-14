#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)

np.random.seed(42)

# 1. Load dataset (relative path)
df_project = pd.read_csv("Telco Customer Churn Project\data\Telco-Customer-Churn.csv")   

# Define your features
TARGET = ['Churn']            
PROJECT_NUMERIC      = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
PROJECT_CATEGORICAL  = ['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges']

# 2. Separate features and target
X_proj = df_project.drop(columns=['Churn'])
y_proj = df_project['Churn']

# 3. Train/test split
X_proj_train, X_proj_test, y_proj_train, y_proj_test = train_test_split(
    X_proj, y_proj, test_size=0.2, random_state=42, stratify=y_proj
)

# 4. Preprocessing
proj_numeric_transformer     = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

proj_categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

proj_preprocessor = ColumnTransformer(transformers=[
    ("num", proj_numeric_transformer,     PROJECT_NUMERIC),
    ("cat", proj_categorical_transformer, PROJECT_CATEGORICAL)
])

# 5 Build the pipeline
proj_pipeline = Pipeline(steps=[
    ("preprocessor", proj_preprocessor),
    ("classifier",   RandomForestClassifier(n_estimators=100, random_state=42))
])

proj_pipeline.fit(X_proj_train, y_proj_train)
y_proj_pred = proj_pipeline.predict(X_proj_test)

print(f"Accuracy: {accuracy_score(y_proj_test, y_proj_pred):.4f}")
print()
print(classification_report(y_proj_test, y_proj_pred))

# 6 Evaluate the model
cm = confusion_matrix(y_proj_test, y_proj_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Confusion Matrix — {TARGET} prediction")
plt.tight_layout()
plt.show()

# F3b: Calculate and print AUC-ROC
y_proj_proba = proj_pipeline.predict_proba(X_proj_test)[:, 1]
auc = roc_auc_score(y_proj_test, y_proj_proba)
print(f"AUC-ROC: {auc:.4f}")

# F3c: Run 5-fold cross-validation
cv = cross_val_score(proj_pipeline, X_proj_train, y_proj_train, cv=5, scoring="roc_auc")
print(f"CV AUC:  {cv.mean():.4f} ± {cv.std():.4f}")
