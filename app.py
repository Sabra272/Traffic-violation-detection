import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ========== Load saved model ==========
with open("traffic_violation_model.pkl", "rb") as f:
    model = pickle.load(f)

# ========== Load dataset to get categories ==========
df = pd.read_csv("traffic_violations_dataset.csv")

# Get unique categories from your actual dataset
vehicle_types = sorted(df["Vehicle_Type"].unique().tolist())
location_types = sorted(df["Location"].unique().tolist())

# Create mappings (must match training encoding order)
vehicle_map = {v: i for i, v in enumerate(vehicle_types)}
location_map = {l: i for i, l in enumerate(location_types)}

# Try loading saved scaler (if available)
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = StandardScaler()
    scaler.fit(df[["Driver_Age", "Fine_Amount"]])  # fit on dataset as fallback

# Try loading saved label encoder (if available)
try:
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except:
    le = LabelEncoder()
    le.fit(df["Violation_Type"])  # fit directly from dataset as fallback

# ========== Streamlit UI ==========
st.title("🚦 Traffic Violation Prediction System")

st.write("Fill in the details below to predict the type of traffic violation:")

# Input fields
age = st.number_input("Driver Age", min_value=18, max_value=100, value=30)
fine = st.number_input("Fine Amount", min_value=0, max_value=10000, value=200)
vehicle = st.selectbox("Vehicle Type", vehicle_types)
location = st.selectbox("Location", location_types)

if st.button("Predict Violation"):
    # Encode categorical inputs
    vehicle_encoded = vehicle_map[vehicle]
    location_encoded = location_map[location]

    # Create input dataframe
    X_input = pd.DataFrame([[age, fine, vehicle_encoded, location_encoded]],
                           columns=["Driver_Age","Fine_Amount","Vehicle_Encoded","Location_Encoded"])
    
    # Scale numeric values
    X_input[["Driver_Age","Fine_Amount"]] = scaler.transform(X_input[["Driver_Age","Fine_Amount"]])

    # Predict
    pred = model.predict(X_input)[0]
    pred_label = le.inverse_transform([pred])[0]

    st.success(f"Predicted Violation: *{pred_label}*")