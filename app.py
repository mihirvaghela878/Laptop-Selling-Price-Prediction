import streamlit as st
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

st.title("💻 Laptop Selling Price Prediction")
st.write("Enter laptop specifications to predict the selling price")

# Inputs
ssd = st.number_input("SSD (GB)", min_value=0)
hdd = st.number_input("HDD (GB)", min_value=0)
ram = st.number_input("RAM (GB)", min_value=1)

resolution_width = st.number_input("Resolution Width", min_value=800)
resolution_height = st.number_input("Resolution Height", min_value=600)

cpu_name = st.selectbox(
    "CPU Name",
    [
        "Intel Core i3",
        "Intel Core i5",
        "Intel Core i7",
        "AMD Ryzen 5",
        "AMD Ryzen 7"
    ]
)

cpu_speed = st.number_input("CPU Speed (GHz)", min_value=1.0)
inches = st.number_input("Screen Size (inches)", min_value=10.0)
weight_kg = st.number_input("Weight (kg)", min_value=0.5)

ispanel = st.selectbox("IPS Panel", [0, 1])
touchscreen = st.selectbox("Touchscreen", [0, 1])

# Prediction
if st.button("Predict Price"):
    input_data = [[
        ssd,
        ram,
        resolution_width,
        resolution_height,
        cpu_name,
        inches,
        cpu_speed,
        weight_kg,
        hdd,
        ispanel,
        touchscreen
    ]]

    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Laptop Price: ₹ {round(prediction[0], 2)}")
