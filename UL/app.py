# streamlit_app.py
import streamlit as st
import requests

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# Title of the Streamlit app
st.title("Player Prediction")

# Input fields for the user to provide feature values
age = st.number_input("Age", min_value=0, max_value=100, value=25)
appearance = st.number_input("Appearance", min_value=0, max_value=500, value=50)
goals = st.number_input("Goals", min_value=0, max_value=100, value=20)
minutes_played = st.number_input("Minutes Played", min_value=0, max_value=5000, value=1500)
price_category = st.selectbox("Price Category", options=["Premium", "Mid", "Budget"])

# Collect the inputs into a dictionary
input_data = {
    "age": age,
    "appearance": appearance,
    "goals": goals,
    "minutes_played": minutes_played,
    "price_category": price_category
}

# Button to trigger prediction
if st.button("Predict Player Value"):
    # Send the data to the FastAPI backend for prediction
    response = requests.post(FASTAPI_URL, json=input_data)
    
    # Check if the request was successful
    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Predicted Value: {prediction['pred']} Euro")
    else:
        st.write("Error in prediction. Please try again.")
