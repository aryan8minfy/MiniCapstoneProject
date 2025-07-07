from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("best_model.pkl")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = pd.DataFrame([{
            'job_title': request.form['job_title'],
            'experience_level': request.form['experience_level'],
            'employment_type': request.form['employment_type'],
            'company_size': request.form['company_size'],
            'company_location': request.form['company_location'],
            'currency': request.form['currency'],
            'salary_currency': request.form['salary_currency'],
            'remote_ratio': float(request.form['remote_ratio']),
            'years_experience': float(request.form['years_experience']),
            'base_salary': float(request.form['base_salary']),
            'bonus': float(request.form['bonus']),
            'stock_options': float(request.form['stock_options'])
        }])

        prediction = model.predict(input_data)[0]
        prediction_text = f"💰 Predicted Adjusted Salary: ${prediction:,.2f}"
        return render_template('index.html', prediction_text=prediction_text)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
