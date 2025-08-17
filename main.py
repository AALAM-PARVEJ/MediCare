"""
MediCare Web App
Slogan: Prediction is better than cure

Backend: Flask + SQLite
Purpose: Symptom-based disease prediction with user profiles.
"""

from flask import Flask, render_template, request, redirect, url_for, session
import joblib, sqlite3, numpy as np, random
import joblib
import numpy as np
import wikipedia

# Load your trained model
model = joblib.load("model_bundle.joblib")


# ==========================================================
# SECTION 1: Flask Setup
# ==========================================================
app = Flask(__name__)
app.secret_key = "secret123"   # Secret key for user session management


# ==========================================================
# SECTION 2: Load Machine Learning Model
# ==========================================================
# model_bundle.joblib contains: svm, rf, nb, scaler, label_encoder, feature_names
bundle = joblib.load("model_bundle.joblib")
feature_names = bundle['feature_names']


# ==========================================================
# SECTION 3: Database Setup
# ==========================================================
# ----------------- Database Setup -----------------
def init_db():
    """
    Initialize SQLite database with tables:
    - users: patient_id, full_name, email, password
    - history: stores predicted diseases for each user
    - feedback: stores user feedback
    If app.db doesn't exist, create it fresh.
    """

    import os
    DB_FILE = "app.db"

    # If DB doesn't exist, create it
    if not os.path.exists(DB_FILE):
        print("⚠️ Database not found. Creating new app.db...")

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        # Users table with full_name + patient_id
        c.execute("""CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            full_name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )""")

        # History table
        c.execute("""CREATE TABLE history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            symptoms TEXT,
            disease TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        # Feedback table
        c.execute("""CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            comment TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.commit()
        conn.close()
        print("✅ New database created successfully with full_name column!")

    else:
        print("ℹ️ Database already exists. Skipping creation.")

# Run DB init once at server start
init_db()


# ==========================================================
# SECTION 4: Helper Functions
# ==========================================================
def generate_patient_id():
    """
    Generate a unique 10-digit Patient ID.
    Example: 1234567890
    """
    return str(random.randint(1000000000, 9999999999))

# ==========================================================
# SECTION 4B: Prediction Helper
# ==========================================================
def model_predict(input_data):
    """
    Predict disease based on symptoms input.
    input_data: list of 0/1 values for symptoms
    Returns: (predicted_disease, confidence)
    """
    # Convert to numpy array and reshape
    X = np.array(input_data).reshape(1, -1)

    # Scale input
    X_scaled = bundle['scaler'].transform(X)

    # Use Random Forest (you can also try svm or nb)
    model = bundle['rf']
    probs = model.predict_proba(X_scaled)[0]

    # Get highest confidence class
    idx = np.argmax(probs)
    disease = bundle['label_encoder'].inverse_transform([idx])[0]
    confidence = probs[idx]

    return disease, confidence



# ==========================================================
# SECTION 5: Routes
# ==========================================================

# ---------- Home Page ----------
@app.route("/")
def home():
    """ Home page of MediCare """
    return render_template("index.html")


# ---------- Login Page ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Login page for users.
    Users can login using either:
    - Patient ID + Password
    - OR Email + Password
    """
    if request.method == "POST":
        identifier = request.form["identifier"]  # could be patient_id or email
        password = request.form["password"]

        conn = sqlite3.connect("app.db")
        c = conn.cursor()
        c.execute("""SELECT patient_id, email FROM users 
                     WHERE (patient_id=? OR email=?) AND password=?""",
                  (identifier, identifier, password))
        user = c.fetchone()
        conn.close()

        if user:
            # Save user session
            session["user"] = user[0]   # patient_id
            session["email"] = user[1]  # email
            return redirect(url_for("profile"))
        else:
            return render_template("login.html", error="❌ Invalid Patient ID/Email or password")

    return render_template("login.html")


# ---------- Signup Page ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    """
    Signup page for new users.
    User enters Email + Password.
    System generates a unique 10-digit Patient ID automatically.
    """
    if request.method == "POST":
        full_name = request.form["full_name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # Check password confirmation
        if password != confirm_password:
            return render_template("signup.html", error="⚠️ Passwords do not match")

        # Generate new patient ID
        patient_id = generate_patient_id()

        try:
            # Insert into database
            conn = sqlite3.connect("app.db")
            c = conn.cursor()
            c.execute("INSERT INTO users (patient_id, full_name, email, password) VALUES (?, ?, ?,?)", 
                      (patient_id, full_name, email, password))
            conn.commit()
            conn.close()

            # Save session and redirect to profile
            session["user"] = patient_id
            session["full_name"] = full_name
            session["email"] = email
            return redirect(url_for("profile"))

        except:
            return render_template("signup.html", error="⚠️ Email already exists. Try another.")

    return render_template("signup.html")


# ---------- Patient Profile ----------
@app.route("/profile")
def profile():
    """
    Patient profile page after login/signup.
    Displays:
    - Patient ID (unique 10-digit number)
    - Email
    - Dashboard buttons (Check Disease, Consultation History, Feedback)
    """
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("profile.html", 
                           patient_id=session["user"], 
                           full_name=session["full_name"],
                           email=session["email"])


# ---------- Logout ----------
@app.route("/logout")
def logout():
    """
    Logout route → clears session and returns to homepage
    """
    session.pop("user", None)
    session.pop("email", None)
    return redirect(url_for("home"))


# ==========================================================

# ---------- Check Disease ----------
# ---------- Check Disease ----------
# Convert underscore symptoms into readable names
def prettify(symptom):
    return symptom.replace("_", " ").title()

all_symptoms = feature_names  # from your trained model

# Pre-define categories (subset of symptoms)
symptom_categories = {
    "Skin Related": [
        "itching", "skin_rash", "nodal_skin_eruptions", "blister", 
        "red_sore_around_nose", "yellow_crust_ooze", "acne", "scurring", "skin_peeling"
    ],
    "Stomach & Digestive": [
        "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting",
        "abdominal_pain", "diarrhoea", "indigestion", "nausea", "constipation"
    ],
    "Respiratory": [
        "cough", "high_fever", "breathlessness", "chest_pain",
        "phlegm", "throat_irritation", "sinus_pressure", "runny_nose"
    ],
    "Neurological & Mental": [
        "headache", "dizziness", "loss_of_balance", "depression", "irritability", "altered_sensorium"
    ]
}

# Collect all already categorized
categorized = set(sum(symptom_categories.values(), []))

# Put remaining into "Other"
remaining = [s for s in all_symptoms if s not in categorized]
symptom_categories["Other"] = remaining

# Prettify names for display
symptom_categories = {
    cat: [prettify(s) for s in lst] for cat, lst in symptom_categories.items()
}


@app.route("/predict", methods=["GET", "POST"])
def predict():
    disease = None
    confidence = None
    if request.method == "POST":
        selected = request.form.getlist("symptoms")
        # Convert selection to model input
        input_data = [1 if prettify(s) in selected else 0 for s in feature_names]
        disease, confidence = model_predict(input_data)

    return render_template("predict.html",
                           symptom_categories=symptom_categories,
                           disease=disease,
                           confidence=confidence)


# SECTION 6: Run Flask
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
