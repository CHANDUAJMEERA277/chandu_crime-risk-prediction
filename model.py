import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("RMS_Crime_Incidents.csv")

# Encode categorical columns
le_type = LabelEncoder()
le_loc = LabelEncoder()
le_day = LabelEncoder()

df["CrimeType"] = le_type.fit_transform(df["CrimeType"])
df["Location"] = le_loc.fit_transform(df["Location"])
df["Day"] = le_day.fit_transform(df["Day"])

X = df[["CrimeType", "Location", "Hour", "Day"]]
y = df["Severity"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("crime_model.pkl", "wb"))
pickle.dump(le_type, open("le_type.pkl", "wb"))
pickle.dump(le_loc, open("le_loc.pkl", "wb"))
pickle.dump(le_day, open("le_day.pkl", "wb"))

print("Model trained and saved successfully")
