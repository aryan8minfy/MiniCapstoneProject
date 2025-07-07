import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

def run_model_training():
    df = pd.read_csv("/opt/airflow/data/processed/cleaned.csv")
    TARGET = 'adjusted_total_usd'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
    }

    best_score = -float("inf")
    best_pipeline = None
    best_name = None

    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment("salary_prediction_pipeline")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(pipeline, name)

            if r2 > best_score:
                best_score = r2
                best_pipeline = pipeline
                best_name = name

    os.makedirs("/opt/airflow/models", exist_ok=True)
    joblib.dump(best_pipeline, "/opt/airflow/models/best_model.pkl")
