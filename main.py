"""
MediCare Web App
Slogan: Prediction is better than cure

Backend: Flask + SQLite
Purpose: Symptom-based disease prediction with user profiles.
"""

from flask import Flask, render_template, request, redirect, url_for, session
import joblib, sqlite3, numpy as np

# ----------------- Flask App Setup -----------------
app = Flask(__name__)
app.secret_key = "secret123"  # Used for session cookies (like user login sessions)

# ----------------- Load ML Model -----------------
# model_bundle.joblib contains: svm, rf, nb, scaler, label_encoder, feature_names
bundle = joblib.load("model_bundle.joblib")
feature_names = bundle['feature_names']


# ----------------- Database Setup -----------------
def init_db():
    """
    Initialize SQLite database with tables:
    - users: store username & password
    - history: stores predicted diseases for each user
    - feedback: stores user feedback
    """
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    # Create Users table
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )""")

    # Create Consultation History table
    c.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        symptoms TEXT,
        disease TEXT,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # Create Feedback table
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        comment TEXT,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    conn.commit()
    conn.close()

# Run once at start
init_db()


# ----------------- Routes -----------------

@app.route("/")
def home():
    """
    Home page of MediCare.
    Shows navbar + Welcome message + Get Started button.
    """
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Login page for existing users.
    Checks username & password from database.
    If valid → redirect to profile.
    """
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("app.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect(url_for("profile"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """
    Signup page for new users.
    Saves username & password in database.
    After signup → redirect to profile.
    """
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        try:
            conn = sqlite3.connect("app.db")
            c = conn.cursor()
            c.execute("INSERT INTO users(username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()

            session["user"] = username
            return redirect(url_for("profile"))

        except:
            return render_template("signup.html", error="⚠️ Username already exists. Try another.")

    return render_template("signup.html")


@app.route("/profile")
def profile():
    """
    Patient profile page after login/signup.
    Shows user dashboard with options:
    - Check Disease
    - Consultation History
    - Feedback
    """
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("profile.html", user=session["user"])


@app.route("/logout")
def logout():
    """
    Logout route → clears session & back to home page.
    """
    session.pop("user", None)
    return redirect(url_for("home"))


# ----------------- Run Flask -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
