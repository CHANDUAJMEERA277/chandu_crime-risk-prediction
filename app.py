from flask import Flask, render_template, request
import pickle
import csv
import os

app = Flask(__name__)

# ---------------- LOAD MODEL & ENCODERS ----------------
model = pickle.load(open("crime_model.pkl", "rb"))
le_type = pickle.load(open("le_type.pkl", "rb"))
le_loc = pickle.load(open("le_loc.pkl", "rb"))
le_day = pickle.load(open("le_day.pkl", "rb"))

# ---------------- CITY → ZONE MAPPING ------------------
CITY_TO_ZONE = {
    "Hyderabad": "Area1",
    "Warangal": "Area1",
    "Khammam": "Area1",
    "Mahabubabad": "Area1",
    "Delhi": "Area2",
    "Mumbai": "Area3",
    "Gujarat": "Area3"
}

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # -------- READ FORM DATA --------
        crime_type = request.form["crime_type"]
        city = request.form["location"]
        date_str = request.form["date"]
        time_str = request.form["time"]     # HH:MM
        day = request.form["day"]

        # -------- PROCESS INPUT --------
        zone = CITY_TO_ZONE.get(city, "Area1")
        hour = int(time_str.split(":")[0])

        ct = le_type.transform([crime_type])[0]
        loc = le_loc.transform([zone])[0]
        d = le_day.transform([day])[0]

        # -------- PREDICTION --------
        pred = model.predict([[ct, loc, hour, d]])[0]
        prediction = "High Risk" if pred >= 4 else "Low Risk"

        # -------- SAVE REPORT --------
        file_exists = os.path.isfile("crime_reports.csv")

        with open("crime_reports.csv", "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "CrimeType",
                    "City",
                    "Zone",
                    "Date",
                    "Time",
                    "Day",
                    "Prediction"
                ])

            writer.writerow([
                crime_type,
                city,
                zone,
                date_str,
                time_str,
                day,
                prediction
            ])

    return render_template("index.html", prediction=prediction)

# ---------------- RUN SERVER (LOCAL + CLOUD SAFE) ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
