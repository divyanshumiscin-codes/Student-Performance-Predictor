# Student Performance Predictor

A web app that predicts whether a student will pass or fail, based on their study habits, attendance, and past marks. Built as a 5th semester Diploma project (Computer Engineering, Sigma University, Vadodara).

## What it does

You enter a student's details — how many hours they study per week, how many classes they've missed, past failures, and their last two exam marks — and the app tells you:

- Will they pass or fail
- Their expected percentage
- How risky their situation is (Low / Medium / High)
- A plain explanation of why the prediction came out this way

It uses a Random Forest model trained on real student data (the UCI Student Performance dataset) to make these predictions.

## Features

- Predict pass/fail and expected percentage for a student
- See which factors mattered most in a prediction
- Compare a student's marks to the class average
- Compare the same student's chances in Math vs Language
- Try different numbers with live sliders and watch the prediction update instantly
- Compare two different ML models (Random Forest vs Logistic Regression) side by side
- Upload a CSV to predict for many students at once
- Keep a history of past predictions, and export it as CSV
- Separate views for Teachers and Students

## Tech used

Python, Flask, scikit-learn, pandas, SQLite, Chart.js, Bootstrap. Models were trained in Google Colab.

## Where the data came from

UCI's Student Performance dataset — combines Math and Portuguese (Language) subject records, about 1,044 students in total. The trained models and data are in the `MODEL` folder.

## How well the models perform

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Accuracy | 91.39% | 87.56% |
| Precision | 91.72% | 87.73% |
| Recall | 96.64% | 95.97% |
| F1-score | 94.12% | 91.67% |

Random Forest is the one actually used for predictions — Logistic Regression is there for comparison.

## How to run it

1. Clone this repo
2. Create a virtual environment and activate it:

python -m venv venv
venv\Scripts\activate

3. Install the required packages:

pip install -r requirements.txt

4. Set up the database (only needed the first time):    

python database.py

5. Run the app:

python app.py

6. Open `http://127.0.0.1:5000` in your browser

## Project layout

student-performance-predictor/
├── app.py -> Flask app, all the routes and prediction logic
├── database.py -> sets up the SQLite database
├── MODEL/ -> trained models, scaler, and training data
├── static/style.css -> styling
├── templates/ -> all the HTML pages
└── requirements.txt