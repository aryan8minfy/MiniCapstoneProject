# Instilit – Scalable Global Salary Intelligence System

This repository contains the capstone project **Instilit**, an end-to-end scalable AI platform to predict total compensation for software professionals across roles, locations, and experience levels. It demonstrates the use of machine learning, MLOps tooling, and modern deployment workflows.

---

## 🎯 Problem Statement

Multinational HR analytics firm **Instilit** aims to deliver **data-backed salary benchmarks** to help employers make equitable compensation decisions globally. The system predicts **total salary packages (base + bonus + stock)** while handling inconsistencies and variations in job titles and geographies.

---

## 🗂️ Dataset

**File:** `Software_Salaries.csv`

**Features include:**

- `job_title`
- `experience_level`
- `employment_type`
- `company_size`
- `company_location`
- `remote_ratio`
- `salary_currency`
- `years_experience`
- `base_salary`
- `bonus`
- `stock_options`
- `total_salary`
- `salary_in_usd`
- `currency`
- `conversion_rate`
- `adjusted_total_usd`

Data was cleaned to handle:

- Typos in job titles
- Missing values in experience level and employment type
- Outliers in numeric features

---

## ⚙️ Project Workflow

### 1️⃣ Data Cleaning & Preprocessing
- Fixed inconsistent job titles using mapping dictionaries.
- Removed duplicates and handled missing values.
- Outliers were removed using IQR-based filtering.
- Features and target (`adjusted_total_usd`) were split.

---

### 2️⃣ Model Training

**Algorithms Used:**
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

All models were trained in pipelines combining preprocessing (OneHotEncoder + StandardScaler) and regressors.

**Evaluation Metrics:**
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

✅ **Screenshot to add here:**
*Model performance comparison table*
<img width="938" alt="Comparisionimage" src="https://github.com/user-attachments/assets/abb66b70-e4b3-4c73-a7a5-50232735d36f" />


---

### 3️⃣ Model Tracking with MLflow
- All experiments, parameters, metrics, and models logged with MLflow.
- The best model was automatically selected based on R² and saved as `best_model.pkl`.

✅ **Screenshot to add here:**
*MLflow UI showing runs and metrics*
<img width="954" alt="Registeredmodel" src="https://github.com/user-attachments/assets/67f4576f-ef9e-496d-93f4-22d53829065f" />


---

### 4️⃣ Explainability with SHAP
- SHAP values generated to interpret feature contributions to predictions.
- Beeswarm plots used for visualizing impact.

✅ **Screenshot to add here:**
*SHAP beeswarm plot*
![Shap](https://github.com/user-attachments/assets/d08055a9-4fea-4268-b89a-04337a15614c)


---

### 5️⃣ Drift Detection with Evidently AI
Three drift reports were generated:
- **Data Drift Report:** Changes in feature distributions.
- **Model Drift Report:** Changes in target distributions.
- **Concept Drift Report:** Changes in prediction vs. actual relationship.

Reports were saved as HTML.

✅ **Screenshot to add here:**
*Evidently data drift report*
`<img width="955" alt="image" src="https://github.com/user-attachments/assets/88ed6ba0-7aaa-4db5-a2a4-d0315ae6a2c9" />


---

### 6️⃣ Batch Predictions with Streamlit
- A **Streamlit UI** lets users upload a CSV file and see salary predictions.
- Predictions are displayed in a table and can be downloaded as a CSV.
- `adjusted_total_usd` column is excluded from the output.

✅ **Screenshot to add here:**
*Streamlit app file upload and predictions*
<img width="958" alt="StreamlitOutputImage" src="https://github.com/user-attachments/assets/0a1255bd-10af-4baf-ab86-9f0c13cc4fd6" />


---

### 7️⃣ REST API with Flask
- A **Flask API** was developed to serve the trained model.
- Accepts JSON payloads and returns predicted salaries.
- Allows integration with external HR systems.

✅ **Screenshot to add here:**
*Flask API response*
<img width="957" alt="Flaskimagecapstone" src="https://github.com/user-attachments/assets/7e9fc0c0-cbb1-4092-99e2-14adcd9fefbe" />


---

### 8️⃣ Workflow Orchestration with Airflow
- An **Apache Airflow DAG** automates:
  - Data loading
  - Model retraining
  - Drift report generation
  - Model versioning
---

## 🧩 Technologies Used

- **Python 3.x**
- **pandas**, **scikit-learn**, **xgboost**
- **SHAP** for explainability
- **MLflow** for experiment tracking
- **Evidently AI** for drift detection
- **Flask** for API deployment
- **Streamlit** for user-facing UI
- **Apache Airflow** for orchestration
- **PostgreSQL** for data persistence

---
