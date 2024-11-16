import streamlit as st
import requests

# FastAPI URL for prediction endpoint
api_url = "https://api-nx09.onrender.com/predict"

# Streamlit UI
st.title("Football Player Prediction Model")
st.write("Welcome to the prediction app! Enter the details below to make a prediction.")

# Input fields for the user
age = st.number_input("Age", min_value=18, max_value=40, step=1)
appearance = st.number_input("Appearance", min_value=0, max_value=500, step=1)
goals = st.number_input("Goals", min_value=0, max_value=300, step=1)
minutes_played = st.number_input("Minutes Played", min_value=0, max_value=50000, step=1)
highest_valuated_price_euro = st.number_input("Highest Valuated Price (Euro)", min_value=0.0, step=0.1)
price_category = st.selectbox("Price Category", options=["Premium", "Mid", "Budget"])

# When the user clicks on the "Predict" button
if st.button("Predict"):
    # Prepare the data to be sent to the FastAPI backend
    input_data = {
        "age": age,
        "appearance": appearance,
        "goals": goals,
        "minutes_played": minutes_played,
        "Highest_valuated_price_euro": highest_valuated_price_euro,
        "price_category": price_category
    }
    
    # Send a POST request to the FastAPI prediction endpoint
    response = requests.post(api_url, json=input_data)
    
    if response.status_code == 200:
        # If the prediction is successful, display the result
        prediction = response.json()["pred"]
        st.write(f"Prediction: {prediction}")
    else:
        # Handle errors (e.g., if the API is down)
        st.error(f"Error: {response.status_code}. Unable to get prediction.")
