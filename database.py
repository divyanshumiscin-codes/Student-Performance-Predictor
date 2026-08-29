import sqlite3

def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            subject TEXT NOT NULL,
            study_hours REAL NOT NULL,
            attendance_issues INTEGER NOT NULL,
            past_failures INTEGER NOT NULL,
            previous_marks_1 REAL NOT NULL,
            previous_marks_2 REAL NOT NULL,
            predicted_result TEXT NOT NULL,
            confidence REAL NOT NULL,
            predicted_percentage REAL NOT NULL,
            risk_category TEXT NOT NULL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

if __name__ == '__main__':
    init_db()