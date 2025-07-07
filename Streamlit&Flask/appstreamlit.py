import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.title("💼 Software Salary Predictor")

# CSV File Uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file containing job data", type=["csv"])

if uploaded_file is not None:
    # Read CSV into DataFrame
    input_df = pd.read_csv(uploaded_file)
    st.write("✅ File uploaded successfully. Here is a preview:")
    st.dataframe(input_df.head())

    # Validate required columns
    required_columns = [
        "job_title",
        "experience_level",
        "employment_type",
        "company_size",
        "company_location",
        "currency",
        "salary_currency",
        "remote_ratio",
        "years_experience",
        "base_salary",
        "bonus",
        "stock_options"
    ]

    missing_cols = [col for col in required_columns if col not in input_df.columns]
    if missing_cols:
        st.error(f"❌ The following columns are missing in your CSV: {missing_cols}")
    else:
        if st.button("Predict Salaries"):
            # Predict
            predictions = model.predict(input_df)

            # Remove adjusted_total_usd if it exists
            results_df = input_df.drop(
                columns=[col for col in ["adjusted_total_usd"] if col in input_df.columns]
            ).copy()

            # Add predictions
            results_df["Predicted_Adjusted_Total_Salary"] = predictions

            st.success("✅ Predictions completed!")
            st.dataframe(results_df)

            # Optionally, let users download results
            csv_output = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions as CSV",
                data=csv_output,
                file_name="predicted_salaries.csv",
                mime="text/csv"
            )
else:
    st.info("👈 Please upload a CSV file to get started.")
