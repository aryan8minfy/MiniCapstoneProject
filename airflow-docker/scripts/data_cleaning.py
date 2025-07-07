import pandas as pd
import os

def run_cleaning():
    df = pd.read_csv("/opt/airflow/data/Software_Salaries.csv")

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
    df.drop(columns=['education', 'skills', 'total_salary', 'salary_in_usd', 'conversion_rate'], inplace=True)
    df = clean_job_titles(df)
    df['experience_level'].fillna(df['experience_level'].mode()[0], inplace=True)
    df['employment_type'].fillna(df['employment_type'].mode()[0], inplace=True)

    os.makedirs("/opt/airflow/data/processed", exist_ok=True)
    df.to_csv("/opt/airflow/data/processed/cleaned.csv", index=False)
