from flask import Flask, render_template, request, session, redirect, url_for, Response
import joblib
import sqlite3
import csv
import csv
import pandas as pd
import io

app = Flask(__name__)
app.secret_key = 'spp-dev-secret-key'  # fine for a local college demo; a real deployment would use an environment variable

classifier = joblib.load('MODEL/classifier_model.pkl')
regressor = joblib.load('MODEL/regressor_model.pkl')

FEATURE_IMPORTANCE = {}
with open('MODEL/feature_importance.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        FEATURE_IMPORTANCE[row['feature']] = float(row['importance'])
def compute_class_averages():
    df_math = pd.read_csv('MODEL/student-mat.csv', sep=';')
    df_por = pd.read_csv('MODEL/student-por.csv', sep=';')
    avg_math = (df_math['G3'] / 20 * 100).mean()
    avg_por = (df_por['G3'] / 20 * 100).mean()
    return {0: round(avg_math, 2), 1: round(avg_por, 2)}

CLASS_AVERAGE = compute_class_averages()


def generate_explanation(result, subject_name, marks1, marks2, attendance, failures, study_hours, risk_category):
    factors = []

    if marks2 < 10:
        factors.append(('previous_marks_2', f"low recent marks in {subject_name} ({marks2}/20)"))
    elif marks2 >= 15:
        factors.append(('previous_marks_2', f"strong recent marks in {subject_name} ({marks2}/20)"))
    else:
        factors.append(('previous_marks_2', f"moderate recent marks ({marks2}/20)"))

    if marks1 < 10:
        factors.append(('previous_marks_1', f"low earlier marks ({marks1}/20)"))
    elif marks1 >= 15:
        factors.append(('previous_marks_1', f"strong earlier marks ({marks1}/20)"))
    else:
        factors.append(('previous_marks_1', f"moderate earlier marks ({marks1}/20)"))

    if attendance >= 8:
        factors.append(('attendance_issues', f"a high number of absences ({attendance})"))
    elif attendance <= 2:
        factors.append(('attendance_issues', f"consistent attendance ({attendance} absences)"))
    else:
        factors.append(('attendance_issues', f"a moderate number of absences ({attendance})"))

    if failures > 0:
        factors.append(('past_failures', f"{failures} past failure(s) in earlier terms"))
    else:
        factors.append(('past_failures', "no past failures"))

    if study_hours <= 1:
        factors.append(('study_hours', "low weekly study hours"))
    elif study_hours >= 3:
        factors.append(('study_hours', "strong weekly study habits"))
    else:
        factors.append(('study_hours', "moderate weekly study habits"))

    factors.sort(key=lambda item: FEATURE_IMPORTANCE.get(item[0], 0), reverse=True)
    reason_text = "; ".join(text for _, text in factors)

    recommendation = {
        'Low': "No intervention needed right now — encourage the student to maintain this pattern.",
        'Medium': "Worth monitoring — a short check-in on study habits or attendance could help before it slips further.",
        'High': "Recommended for early intervention — a teacher follow-up on attendance and revision support is advised."
    }.get(risk_category, "")

    return (f"This student is predicted to {result.lower()} in {subject_name}, with {risk_category.lower()} risk, "
            f"primarily shaped by (ordered by the model's own feature importance): {reason_text}. {recommendation}")


@app.route('/')
def home():
    return render_template('role_select.html')


@app.route('/set-role/<role>')
def set_role(role):
    if role not in ('teacher', 'student'):
        return redirect(url_for('home'))
    session['role'] = role
    return redirect(url_for('predict_form'))


@app.route('/logout-role')
def logout_role():
    session.pop('role', None)
    return redirect(url_for('home'))


@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    if 'role' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        student_name = request.form.get('student_name', '').strip()
        subject = int(request.form.get('subject'))
        study_hours = float(request.form.get('study_hours'))
        attendance_issues = int(request.form.get('attendance_issues'))
        past_failures = int(request.form.get('past_failures'))
        previous_marks_1 = float(request.form.get('previous_marks_1'))
        previous_marks_2 = float(request.form.get('previous_marks_2'))

        errors = []
        if not student_name:
            errors.append("Student name is required.")
        if not (0 <= previous_marks_1 <= 20):
            errors.append("Previous Marks 1 must be between 0 and 20.")
        if not (0 <= previous_marks_2 <= 20):
            errors.append("Previous Marks 2 must be between 0 and 20.")
        if not (1 <= study_hours <= 4):
            errors.append("Study hours must be between 1 and 4.")
        if not (0 <= past_failures <= 4):
            errors.append("Past failures must be between 0 and 4.")
        if attendance_issues < 0:
            errors.append("Attendance issues cannot be negative.")

        if errors:
            return render_template('predict_form.html', errors=errors)

        input_data = [[study_hours, attendance_issues, past_failures,
                        previous_marks_1, previous_marks_2, subject]]

        predicted_result = classifier.predict(input_data)[0]
        confidence = round(classifier.predict_proba(input_data).max() * 100, 2)
        predicted_percentage = round(regressor.predict(input_data)[0], 2)

        if predicted_percentage >= 75:
            risk_category = "Low"
        elif predicted_percentage >= 50:
            risk_category = "Medium"
        else:
            risk_category = "High"

        subject_name = "Math" if subject == 0 else "Language"
        explanation = generate_explanation(
            predicted_result, subject_name, previous_marks_1, previous_marks_2,
            attendance_issues, past_failures, study_hours, risk_category
        )

        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions
            (student_name, subject, study_hours, attendance_issues, past_failures,
             previous_marks_1, previous_marks_2, predicted_result, confidence,
             predicted_percentage, risk_category, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (student_name, subject_name, study_hours, attendance_issues, past_failures,
              previous_marks_1, previous_marks_2, predicted_result, confidence,
              predicted_percentage, risk_category, explanation))
        conn.commit()
        conn.close()

        return render_template('result.html',
                                student_name=student_name,
                                subject=subject_name,
                                predicted_result=predicted_result,
                                confidence=confidence,
                                predicted_percentage=predicted_percentage,
                                risk_category=risk_category,
                                explanation=explanation,
                                feature_importance=FEATURE_IMPORTANCE,
                                class_average=CLASS_AVERAGE[subject])

    return render_template('predict_form.html', errors=None)

@app.route('/history')
def history():
    if session.get('role') != 'teacher':
        return redirect(url_for('predict_form'))
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY created_at DESC')
    records = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=records)

@app.route('/history/export')
def export_history():
    if session.get('role') != 'teacher':
        return redirect(url_for('predict_form'))

    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY created_at DESC')
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()

    def generate():
        yield ','.join(columns) + '\n'
        for row in rows:
            values = [str(v).replace(',', ';') for v in row]
            yield ','.join(values) + '\n'

    return Response(
        generate(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=prediction_history.csv'}
    )

@app.route('/bulk', methods=['GET', 'POST'])
def bulk_upload():
    if session.get('role') != 'teacher':
        return redirect(url_for('predict_form'))

    results = []
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if file:
            import csv as csv_module
            stream = io.StringIO(file.stream.read().decode('utf-8'))
            reader = csv_module.DictReader(stream)

            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()

            for row in reader:
                student_name = row['student_name'].strip()
                subject = int(row['subject'])
                study_hours = float(row['study_hours'])
                attendance_issues = int(row['attendance_issues'])
                past_failures = int(row['past_failures'])
                previous_marks_1 = float(row['previous_marks_1'])
                previous_marks_2 = float(row['previous_marks_2'])

                input_data = [[study_hours, attendance_issues, past_failures,
                                previous_marks_1, previous_marks_2, subject]]

                predicted_result = classifier.predict(input_data)[0]
                confidence = round(classifier.predict_proba(input_data).max() * 100, 2)
                predicted_percentage = round(regressor.predict(input_data)[0], 2)

                if predicted_percentage >= 75:
                    risk_category = "Low"
                elif predicted_percentage >= 50:
                    risk_category = "Medium"
                else:
                    risk_category = "High"

                subject_name = "Math" if subject == 0 else "Language"
                explanation = generate_explanation(
                    predicted_result, subject_name, previous_marks_1, previous_marks_2,
                    attendance_issues, past_failures, study_hours, risk_category
                )

                cursor.execute('''
                    INSERT INTO predictions
                    (student_name, subject, study_hours, attendance_issues, past_failures,
                     previous_marks_1, previous_marks_2, predicted_result, confidence,
                     predicted_percentage, risk_category, explanation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (student_name, subject_name, study_hours, attendance_issues, past_failures,
                      previous_marks_1, previous_marks_2, predicted_result, confidence,
                      predicted_percentage, risk_category, explanation))

                results.append({
                    'student_name': student_name, 'subject': subject_name,
                    'predicted_result': predicted_result, 'confidence': confidence,
                    'predicted_percentage': predicted_percentage, 'risk_category': risk_category
                })

            conn.commit()
            conn.close()

    return render_template('bulk_upload.html', results=results)
if __name__ == '__main__':
    app.run(debug=True)