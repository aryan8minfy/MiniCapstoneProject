import joblib
import pandas as pd
import shap

def run_shap_explainer():
    model = joblib.load("/opt/airflow/models/best_model.pkl")
    df = pd.read_csv("/opt/airflow/data/processed/cleaned.csv")
    X = df.drop(columns=["adjusted_total_usd"])

    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        explainer = shap.Explainer(model.named_steps['regressor'])
        X_transformed = model.named_steps['preprocessor'].transform(X)
        shap_values = explainer(X_transformed[:100])
        shap.plots.beeswarm(shap_values)
    else:
        print("SHAP only supports tree models.")
