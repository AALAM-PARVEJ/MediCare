"""
MediCare Web App
Slogan: Prediction is better than cure

Backend: Flask + SQLite
Purpose: Symptom-based disease prediction with user profiles.
"""

from flask import Flask, render_template, request, redirect, url_for, session
import joblib, sqlite3, numpy as np, random, wikipedia, secrets
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# ---------------- EMAIL CONFIG -----------------
EMAIL_ADDRESS = "aalamp140@gmail.com"
EMAIL_PASSWORD = "yxqu aukw eqez fiom"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ---------------- FLASK SETUP -----------------
app = Flask(__name__)
app.secret_key = "eie7077"

# ---------------- LOAD MODEL -----------------
bundle = joblib.load("model_bundle.joblib")
feature_names = bundle['feature_names']

# ---------------- DATABASE SETUP -----------------
def init_db():
    import os
    DB_FILE = "app.db"
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            full_name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            is_temp INTEGER DEFAULT 0
        )""")
        c.execute("""CREATE TABLE history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            symptoms TEXT,
            disease TEXT,
            confidence REAL,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""CREATE TABLE feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            email TEXT,
            full_name TEXT,
            comment TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
        conn.close()
init_db()

# ---------------- HELPER FUNCTIONS -----------------
def generate_patient_id():
    return str(random.randint(1000000000, 9999999999))

def send_temp_password(to_email, temp_password):
    subject = "MediCare Temporary Password Reset"
    body = f"Hello,\n\nYour temporary password is: {temp_password}\n\n- MediCare Team"
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.send_message(msg)
    server.quit()

def model_predict(input_data):
    X = np.array(input_data).reshape(1, -1)
    X_scaled = bundle['scaler'].transform(X)
    model = bundle['rf']
    probs = model.predict_proba(X_scaled)[0]
    idx = np.argmax(probs)
    disease = bundle['label_encoder'].inverse_transform([idx])[0]
    confidence = probs[idx]
    return disease, confidence

def prettify(symptom):
    return symptom.replace("_", " ").title()

# ---------------- SYMPTOM CATEGORIES -----------------
symptom_categories = {
    "General & Constitutional": ["Fatigue", "Weight Gain", "Weight Loss", "Restlessness", "Lethargy",
                                 "Malaise", "Obesity", "Excessive Hunger", "Increased Appetite",
                                 "Sweating", "Chills", "Shivering", "High Fever", "Mild Fever",
                                 "Toxic Look (Typhos)"],
    "Head, Brain & Neurological": ["Headache", "Dizziness", "Loss Of Balance", "Unsteadiness",
                                   "Weakness Of One Body Side", "Altered Sensorium", "Coma",
                                   "Irritability", "Depression", "Anxiety", "Mood Swings",
                                   "Lack Of Concentration", "Slurred Speech", "Spinning Movements",
                                   "Memory Loss"],
    "Eye Related": ["Pain Behind The Eyes", "Blurred And Distorted Vision",
                    "Visual Disturbances", "Redness Of Eyes", "Watering From Eyes",
                    "Yellowing Of Eyes", "Sunken Eyes"],
    "Ear, Nose & Throat": ["Continuous Sneezing", "Patchy Throat", "Throat Irritation",
                           "Runny Nose", "Congestion", "Sinus Pressure", "Loss Of Smell",
                           "Drying And Tingling Lips", "Red Sore Around Nose"],
    "Respiratory & Chest": ["Cough", "Phlegm", "Mucoid Sputum", "Rusty Sputum", "Blood In Sputum",
                            "Breathlessness", "Chest Pain", "Palpitations", "Fast Heart Rate",
                            "Wheezing"],
    "Digestive & Abdominal": ["Stomach Pain", "Acidity", "Indigestion", "Nausea", "Vomiting",
                              "Loss Of Appetite", "Abdominal Pain", "Diarrhoea", "Constipation",
                              "Belly Pain", "Distention Of Abdomen", "Stomach Bleeding",
                              "Fluid Overload", "Passage Of Gases", "Swelling Of Stomach"],
    "Liver & Urinary": ["Dark Urine", "Yellow Urine", "Acute Liver Failure", "Dehydration",
                        "Burning Micturition", "Spotting Urination", "Frequent Urination",
                        "Bladder Discomfort", "Foul Smell Of Urine",
                        "Continuous Feel Of Urine", "Polyuria", "Painful Urination",
                        "Blood In Urine"],
    "Skin, Hair & Nails": ["Itching", "Skin Rash", "Nodal Skin Eruptions", "Internal Itching",
                           "Dischromic Patches", "Pus Filled Pimples", "Blackheads", "Scurring",
                           "Skin Peeling", "Silver Like Dusting", "Small Dents In Nails",
                           "Inflammatory Nails", "Blister", "Yellow Crust Ooze",
                           "Ulcers On Tongue", "Bruising", "Brittle Nails", "Pale Skin"],
    "Musculoskeletal & Joints": ["Joint Pain", "Knee Pain", "Hip Joint Pain", "Back Pain", "Neck Pain",
                                 "Stiff Neck", "Cramps", "Muscle Weakness", "Muscle Wasting",
                                 "Movement Stiffness", "Swelling Joints", "Swollen Legs",
                                 "Swollen Blood Vessels", "Swollen Extremities", "Painful Walking",
                                 "Bone Pain"],
    "Reproductive & Endocrine": ["Abnormal Menstruation", "Enlarged Thyroid", "Irregular Sugar Level",
                                 "Family History", "History Of Alcohol Consumption",
                                 "Receiving Blood Transfusion", "Receiving Unsterile Injections",
                                 "Extra Marital Contacts"],
    "Lymphatic & Glandular": ["Swelled Lymph Nodes", "Swelling Of Glands", "Puffy Face And Eyes",
                              "Prominent Veins On Calf"],
    "Other Severe Indicators": ["Acute Liver Failure", "Stomach Bleeding", "Coma", "Altered Sensorium"]
}

# ---------------- ROUTES -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        identifier = request.form["identifier"]
        password = request.form["password"]
        user_type = request.form.get("user_type", "user")

        if user_type == "admin":
            if identifier.lower() == "aalamp140@gmail.com" and password == "pass123":
                session["admin"] = True
                return redirect(url_for("admin_dashboard"))
            else:
                return render_template("login.html", error="❌ Admin login failed")

        # Normal user login
        conn = sqlite3.connect("app.db")
        c = conn.cursor()
        c.execute("""SELECT patient_id, full_name, email, password, is_temp 
                     FROM users WHERE patient_id=? OR email=?""", (identifier, identifier))
        user = c.fetchone()
        conn.close()
        if user and user[3] == password:
            session["user"] = user[0]
            session["full_name"] = user[1]
            session["email"] = user[2]
            return redirect(url_for("profile"))
        else:
            return render_template("login.html", error="❌ Invalid user credentials")
    return render_template("login.html")



@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method=="POST":
        full_name=request.form["full_name"]
        email=request.form["email"]
        password=request.form["password"]
        confirm_password=request.form["confirm_password"]
        if password!=confirm_password:
            return render_template("signup.html", error="⚠️ Passwords do not match")
        patient_id=generate_patient_id()
        try:
            conn=sqlite3.connect("app.db")
            c=conn.cursor()
            c.execute("INSERT INTO users (patient_id, full_name, email, password) VALUES (?,?,?,?)",
                      (patient_id, full_name, email, password))
            conn.commit()
            conn.close()
            session["user"]=patient_id
            session["full_name"]=full_name
            session["email"]=email
            return redirect(url_for("profile"))
        except:
            return render_template("signup.html", error="⚠️ Email already exists")
    return render_template("signup.html")

@app.route("/forgot_password", methods=["GET","POST"])
def forgot_password():
    if request.method=="POST":
        email=request.form["email"].strip()
        conn=sqlite3.connect("app.db")
        c=conn.cursor()
        c.execute("SELECT patient_id FROM users WHERE email=?",(email,))
        user=c.fetchone()
        if user:
            temp_password=secrets.token_urlsafe(6)
            c.execute("UPDATE users SET password=?, is_temp=1 WHERE email=?",(temp_password,email))
            conn.commit()
            conn.close()
            try:
                send_temp_password(email,temp_password)
                return render_template("forgot_password.html", message="✅ Temporary password sent")
            except Exception as e:
                return render_template("forgot_password.html", error=f"❌ Failed to send email. {str(e)}")
        else:
            conn.close()
            return render_template("forgot_password.html", error="❌ Email not found")
    return render_template("forgot_password.html")

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    conn=sqlite3.connect("app.db")
    c=conn.cursor()
    c.execute("SELECT is_temp FROM users WHERE patient_id=?",(session["user"],))
    row=c.fetchone()
    temp_flag=row[0]==1 if row else False
    conn.close()
    return render_template("profile.html",
                           patient_id=session["user"],
                           full_name=session["full_name"],
                           email=session["email"],
                           temp_password=temp_flag)

@app.route("/update_password", methods=["POST"])
def update_password():
    if "user" not in session:
        return redirect(url_for("login"))
    new_password=request.form["new_password"]
    confirm_password=request.form["confirm_password"]
    if new_password!=confirm_password:
        return render_template("profile.html",
                               patient_id=session["user"],
                               full_name=session["full_name"],
                               email=session["email"],
                               temp_password=True,
                               error="⚠️ Passwords do not match.")
    conn=sqlite3.connect("app.db")
    c=conn.cursor()
    c.execute("UPDATE users SET password=?, is_temp=0 WHERE patient_id=?",(new_password, session["user"]))
    conn.commit()
    conn.close()
    return render_template("profile.html",
                           patient_id=session["user"],
                           full_name=session["full_name"],
                           email=session["email"],
                           temp_password=False,
                           message="✅ Password updated successfully!")

@app.route("/logout")
def logout():
    session.pop("user",None)
    session.pop("admin",None)
    session.pop("email",None)
    return redirect(url_for("home"))

@app.route("/predict", methods=["GET","POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method=="POST":
        selected=request.form.getlist("symptoms")
        session["selected_symptoms"]=selected
        return redirect(url_for("result"))
    return render_template("predict.html", symptom_categories=symptom_categories,
                           selected=session.get("selected_symptoms",[]))

@app.route("/result")
def result():
    selected=session.get("selected_symptoms",[])
    if not selected:
        return redirect(url_for("predict"))
    input_data=[1 if s in selected else 0 for s in feature_names]
    disease, confidence=model_predict(input_data)
    if "user" in session:
        conn=sqlite3.connect("app.db")
        c=conn.cursor()
        c.execute("INSERT INTO history (patient_id, symptoms, disease, confidence) VALUES (?,?,?,?)",
                  (session["user"], ", ".join(selected), disease, confidence))
        conn.commit()
        conn.close()
    try:
        wiki_summary=wikipedia.summary(disease, sentences=3, auto_suggest=True, redirect=True)
    except:
        wiki_summary=None
    return render_template("result.html",
                           disease=disease,
                           confidence=round(confidence*100,2),
                           wiki=wiki_summary,
                           symptoms=selected)

@app.route("/feedback", methods=["GET","POST"])
def feedback():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method=="POST":
        comment=request.form["comment"]
        patient_id=session["user"]
        full_name=session.get("full_name","")
        email=session.get("email","")
        conn=sqlite3.connect("app.db")
        c=conn.cursor()
        c.execute("INSERT INTO feedback (patient_id,email,full_name,comment) VALUES (?,?,?,?)",
                  (patient_id,email,full_name,comment))
        conn.commit()
        conn.close()
        recipients=["aalamp140@gmail.com"]
        subject="New Feedback from MediCare"
        body=f"Patient ID: {patient_id}\nName: {full_name}\nEmail: {email}\nFeedback: {comment}"
        for recipient in recipients:
            msg=MIMEMultipart()
            msg['From']=EMAIL_ADDRESS
            msg['To']=recipient
            msg['Subject']=subject
            msg.attach(MIMEText(body,'plain'))
            server=smtplib.SMTP(SMTP_SERVER,SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS,EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
        return render_template("thank_you.html", full_name=full_name)
    return render_template("feedback.html")

@app.route("/thank_you")
def thank_you():
    full_name=session.get("full_name","User")
    return render_template("thank_you.html", full_name=full_name)

@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))
    patient_id=session["user"]
    conn=sqlite3.connect("app.db")
    conn.row_factory=sqlite3.Row
    c=conn.cursor()
    c.execute("SELECT ts,symptoms,disease,confidence FROM history WHERE patient_id=? ORDER BY ts DESC",(patient_id,))
    history_data=c.fetchall()
    conn.close()
    return render_template("history.html", history=history_data)

@app.route("/admin")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("login"))
    conn=sqlite3.connect("app.db")
    conn.row_factory=sqlite3.Row
    c=conn.cursor()
    c.execute("SELECT patient_id,full_name,email,comment,ts FROM feedback ORDER BY ts DESC")
    feedbacks=c.fetchall()
    conn.close()
    return render_template("admin_dashboard.html", feedbacks=feedbacks)

# ---------------- RUN SERVER -----------------
if __name__=="__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, port=8080, debug=True)
