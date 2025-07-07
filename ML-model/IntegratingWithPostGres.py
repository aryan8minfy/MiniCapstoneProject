import os
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset

import warnings
warnings.filterwarnings("ignore")

# ============ PostgreSQL Connection ============
user = "postgres"
password = "postgres"
host = "localhost"
port = "5432"
database = "salariesdb"

engine = create_engine(f"postgresql+pg8000://{user}:{password}@{host}:{port}/{database}")

# ============ Load CSV ============
df_csv = pd.read_csv("Software_Salaries.csv")

# ============ Upload CSV to Postgres ============
df_csv.to_sql("software_salaries", engine, if_exists="replace", index=False)
print("✅ CSV uploaded successfully to salariesdb!")

# ============ Load Data Back from Postgres ============
df = pd.read_sql("SELECT * FROM software_salaries", engine)

# ============ Data Cleaning ============
def clean_job_titles(df):
    mapping = {
        'Sofware Engneer': 'Software Engineer',
        'Software Engr': 'Software Engineer',
        'Softwre Engineer': 'Software Engineer',
        'Dt Scientist': 'Data Scientist',
        'Data Scienist': 'Data Scientist',
        'Data Scntist': 'Data Scientist',
        'ML Engr': 'Machine Learning Engineer',
        'ML Enginer': 'Machine Learning Engineer',
        'Machine Learning Engr': 'Machine Learning Engineer',
        'Software Engr': 'Software Engineer',
    }
    df['job_title'] = df['job_title'].replace(mapping)
    df['job_title'] = df['job_title'].str.title().str.strip()
    return df

df.drop_duplicates(inplace=True)

# Drop columns if they exist
drop_cols = ['education', 'skills', 'total_salary', 'salary_in_usd', 'conversion_rate']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

df = clean_job_titles(df)

df['experience_level'] = df['experience_level'].fillna(df['experience_level'].mode()[0])
df['employment_type'] = df['employment_type'].fillna(df['employment_type'].mode()[0])

def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outliers_iqr(df, ['base_salary', 'bonus', 'stock_options', 'adjusted_total_usd'])

# ============ Feature & Target ============
TARGET = 'adjusted_total_usd'
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============ Preprocessing ============
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# ============ Model Setup ============
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
}

mlflow.set_experiment("salary_prediction_pipeline")

best_model = None
best_pipeline = None
best_score = -np.inf
best_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        mlflow.sklearn.log_model(pipeline, name)

        if r2 > best_score:
            best_score = r2
            best_pipeline = pipeline
            best_model = model
            best_name = name

# ============ Save Best Model ============
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "best_model.pkl")
print(f"\n✅ Best model: {best_name} with R²: {best_score:.4f}")

# ============ Evidently: Drift Reports ============
os.makedirs("reports", exist_ok=True)

# 1. Data Drift
data_drift = Report(metrics=[DataDriftPreset()])
data_drift.run(reference_data=X_train, current_data=X_test)
data_drift.save_html("reports/data_drift_report.html")

# 2. Model Drift
model_drift = Report(metrics=[TargetDriftPreset()])
model_drift.run(reference_data=y_train.to_frame(), current_data=y_test.to_frame())
model_drift.save_html("reports/model_drift_report.html")

# 3. Concept Drift
y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)

train_df = X_train.copy()
train_df["target"] = y_train
train_df["prediction"] = y_train_pred

test_df = X_test.copy()
test_df["target"] = y_test
test_df["prediction"] = y_test_pred

concept_drift = Report(metrics=[RegressionPreset()])
concept_drift.run(reference_data=train_df, current_data=test_df)
concept_drift.save_html("reports/concept_drift_report.html")

# ============ SHAP Explainability ============
if best_name in ["XGBoost", "RandomForest"]:
    explainer = shap.Explainer(best_pipeline.named_steps['regressor'])
    transformed_X = best_pipeline.named_steps['preprocessor'].transform(X_test)
    shap_values = explainer(transformed_X[:100])
    shap.plots.beeswarm(shap_values)
else:
    print("SHAP only available for tree-based models.")
