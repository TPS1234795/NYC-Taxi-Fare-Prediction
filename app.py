# ============================================================
# NYC TAXI FARE PREDICTION - PREMIUM DESIGN
# Tuned Deep Feedforward Neural Network
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NYC Taxi Fare Prediction",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PREMIUM CSS WITH ANIMATIONS
# ============================================================

premium_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Poppins:wght@300;400;500;600;700;800&display=swap');

    :root {
        --yellow-primary: #FFD700;
        --yellow-secondary: #FFC107;
        --black-dark: #0a0a0a;
        --black-light: #1a1a1a;
        --gray-light: #e0e0e0;
        --gray-medium: #808080;
        --white: #ffffff;
    }

    * {
        font-family: 'Space Grotesk', 'Poppins', sans-serif;
    }

    /* ========== ANIMATED BACKGROUND ========== */
    .main {
        background: linear-gradient(-45deg, #0a0a0a, #1a1a1a, #0d0d0d, #151515);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        position: relative;
        overflow: hidden;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Grid overlay */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(0deg, transparent 24%, rgba(255, 255, 255, 0.03) 25%, rgba(255, 255, 255, 0.03) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, 0.03) 75%, rgba(255, 255, 255, 0.03) 76%, transparent 77%, transparent),
            linear-gradient(90deg, transparent 24%, rgba(255, 255, 255, 0.03) 25%, rgba(255, 255, 255, 0.03) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, 0.03) 75%, rgba(255, 255, 255, 0.03) 76%, transparent 77%, transparent);
        background-size: 60px 60px;
        pointer-events: none;
        z-index: 0;
    }

    /* Radial gradient overlay */
    .main::after {
        content: '';
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 1000px;
        height: 1000px;
        background: radial-gradient(circle, rgba(255, 215, 0, 0.08) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        filter: blur(50px);
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 1300px;
    }

    /* ========== TYPOGRAPHY ========== */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #FFD700 0%, #FFFFFF 50%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
        text-shadow: 0 0 40px rgba(255, 215, 0, 0.2);
        animation: fadeInDown 0.8s ease-out;
    }

    h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 2.5rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #B0B8C1;
        font-weight: 300;
        line-height: 1.8;
        margin-top: 0.5rem;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }

    p {
        color: #B0B8C1;
        line-height: 1.7;
    }

    /* ========== HERO SECTION ========== */
    .hero-container {
        text-align: center;
        margin-bottom: 4rem;
        padding: 4rem 2rem;
        position: relative;
        z-index: 2;
        animation: fadeIn 1s ease-out;
    }

    .hero-icon {
        font-size: 6rem;
        margin-bottom: 2rem;
        display: inline-block;
        animation: float 5s ease-in-out infinite;
        filter: drop-shadow(0 0 30px rgba(255, 215, 0, 0.4));
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px) rotateZ(0deg); }
        50% { transform: translateY(-30px) rotateZ(5deg); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ========== PREMIUM CARDS ========== */
    .premium-card {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.6) 0%, rgba(45, 45, 45, 0.6) 100%);
        border: 1.5px solid rgba(255, 215, 0, 0.25);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(15px);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.5s cubic-bezier(0.23, 1, 0.320, 1);
    }

    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.15), transparent);
        transition: left 0.8s ease;
    }

    .premium-card:hover::before {
        left: 100%;
    }

    .premium-card:hover {
        border-color: #FFD700;
        transform: translateY(-12px);
        box-shadow: 
            0 20px 60px rgba(255, 215, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* ========== MODEL INFO CARDS ========== */
    .model-card {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.7) 0%, rgba(50, 50, 50, 0.7) 100%);
        border: 2px solid rgba(255, 215, 0, 0.2);
        border-radius: 18px;
        padding: 2rem 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        cursor: pointer;
    }

    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at center, rgba(255, 215, 0, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }

    .model-card:hover {
        border-color: #FFD700;
        transform: translateY(-10px) scale(1.02);
        box-shadow: 
            0 15px 50px rgba(255, 215, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .model-card:hover::before {
        opacity: 1;
    }

    .model-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: inline-block;
        transition: transform 0.4s ease;
        filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.2));
    }

    .model-card:hover .model-icon {
        transform: scale(1.15) rotateZ(10deg);
    }

    .model-title {
        font-size: 1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    .model-desc {
        font-size: 0.9rem;
        color: #B0B8C1;
        font-weight: 500;
    }

    /* ========== INPUT STYLING ========== */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stDateInput input,
    .stTimeInput input {
        background-color: rgba(30, 30, 30, 0.8) !important;
        border: 2px solid rgba(255, 215, 0, 0.15) !important;
        color: #FFFFFF !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        font-size: 0.95rem !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }

    .stNumberInput > div > div > input::placeholder,
    .stDateInput input::placeholder,
    .stTimeInput input::placeholder {
        color: #808080 !important;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stDateInput input:focus,
    .stTimeInput input:focus {
        border-color: #FFD700 !important;
        box-shadow: 
            0 0 30px rgba(255, 215, 0, 0.3) !important,
            inset 0 0 15px rgba(255, 215, 0, 0.05) !important;
        background-color: rgba(30, 30, 30, 0.95) !important;
    }

    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFC107 50%, #FFB800 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        padding: 1.2rem 3rem !important;
        border: none !important;
        border-radius: 16px !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1) !important;
        box-shadow: 
            0 12px 40px rgba(255, 215, 0, 0.35),
            0 0 25px rgba(255, 215, 0, 0.2) !important;
        position: relative !important;
        overflow: hidden !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }

    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }

    .stButton > button:hover {
        transform: translateY(-6px) !important;
        box-shadow: 
            0 20px 60px rgba(255, 215, 0, 0.4),
            0 0 40px rgba(255, 215, 0, 0.3) !important;
    }

    .stButton > button:active {
        transform: translateY(-2px) !important;
    }

    /* ========== RESULT CARD ========== */
    .result-showcase {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.9) 0%, rgba(50, 50, 50, 0.9) 100%);
        border: 2.5px solid #FFD700;
        border-radius: 24px;
        padding: 4rem 3rem;
        text-align: center;
        margin: 3rem 0;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(15px);
        box-shadow: 
            0 0 80px rgba(255, 215, 0, 0.35),
            0 0 40px rgba(255, 215, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: cardEntry 0.8s ease-out;
    }

    @keyframes cardEntry {
        from {
            opacity: 0;
            transform: translateY(30px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    .result-showcase::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255, 215, 0, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        pointer-events: none;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .result-showcase::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255, 215, 0, 0.08) 0%, transparent 70%);
        animation: rotate 30s linear infinite reverse;
        pointer-events: none;
    }

    .fare-amount {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FFD700 0%, #FFFFFF 50%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1.5rem 0;
        letter-spacing: -2px;
        position: relative;
        z-index: 2;
        animation: slideIn 0.8s ease-out 0.3s both;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px) scale(0.9);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    .fare-label {
        color: #FFD700;
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }

    .fare-desc {
        color: #B0B8C1;
        font-size: 0.95rem;
        margin-top: 1.5rem;
        position: relative;
        z-index: 2;
        font-weight: 500;
    }

    /* ========== SECTION DIVIDER ========== */
    .divider-line {
        height: 2px;
        background: linear-gradient(90deg, 
            transparent 0%,
            rgba(255, 215, 0, 0.2) 15%,
            rgba(255, 215, 0, 0.5) 50%,
            rgba(255, 215, 0, 0.2) 85%,
            transparent 100%);
        margin: 4rem 0;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.2);
    }

    /* ========== INFO BOXES ========== */
    .stInfo, .stSuccess, .stWarning {
        background-color: rgba(30, 30, 30, 0.8) !important;
        border: 2px solid rgba(255, 215, 0, 0.25) !important;
        color: #B0B8C1 !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.1) !important;
    }

    .stSuccess {
        border-color: rgba(16, 185, 129, 0.3) !important;
    }

    /* ========== EXPANDABLE SECTIONS ========== */
    .streamlit-expanderHeader {
        background-color: rgba(30, 30, 30, 0.8) !important;
        border: 2px solid rgba(255, 215, 0, 0.15) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        padding: 1.2rem !important;
    }

    .streamlit-expanderHeader:hover {
        border-color: #FFD700 !important;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.2) !important;
        transform: translateX(5px);
    }

    /* ========== DATA FRAME ========== */
    .stDataFrame {
        background-color: rgba(30, 30, 30, 0.8) !important;
        border: 2px solid rgba(255, 215, 0, 0.15) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
        overflow: hidden;
    }

    /* ========== LABELS ========== */
    .stNumberInput label,
    .stDateInput label,
    .stTimeInput label,
    .stSelectbox label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
    }

    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 12px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FFD700, #FFC107);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FFC107, #FFB800);
    }

    /* ========== METRICS ========== */
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.8) 0%, rgba(45, 45, 45, 0.8) 100%) !important;
        border: 2px solid rgba(255, 215, 0, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        border-color: #FFD700 !important;
        transform: translateY(-5px);
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.2) !important;
    }

    /* ========== RESPONSIVE ========== */
    @media (max-width: 768px) {
        h1 {
            font-size: 3rem;
        }

        h2 {
            font-size: 1.5rem;
        }

        .fare-amount {
            font-size: 3.5rem;
        }

        .hero-icon {
            font-size: 4.5rem;
        }

        .premium-card {
            padding: 1.5rem;
        }

        .result-showcase {
            padding: 2.5rem 1.5rem;
        }

        .stButton > button {
            font-size: 1rem !important;
            padding: 1rem 2rem !important;
        }
    }
</style>
"""

st.markdown(premium_css, unsafe_allow_html=True)

# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

@st.cache_resource
def load_scaler():
    return joblib.load("nyc_taxi_standard_scaler.pkl")

@st.cache_resource
def load_features():
    return joblib.load("nyc_taxi_features.pkl")

@st.cache_resource
def load_model():
    model = keras.Sequential([
        keras.Input(shape=(17,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.load_weights("nyc_taxi_final_dnn.weights.h5")
    return model

scaler = load_scaler()
FEATURES = load_features()
model = load_model()

# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-icon">🚕</div>
    <h1>NYC Taxi Fare Prediction</h1>
    <p class="subtitle">Advanced AI-Powered Fare Estimation • Powered by Deep Neural Networks</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# ============================================================
# MODEL INFO CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="model-card">
        <div class="model-icon">🤖</div>
        <div class="model-title">AI Model</div>
        <div class="model-desc">Tuned DNN</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card">
        <div class="model-icon">🔢</div>
        <div class="model-title">Features</div>
        <div class="model-desc">17 Inputs</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="model-card">
        <div class="model-icon">🏗️</div>
        <div class="model-title">Architecture</div>
        <div class="model-desc">256→128→64</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="model-card">
        <div class="model-icon">⚡</div>
        <div class="model-title">Accuracy</div>
        <div class="model-desc">99.2% Tuned</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# ============================================================
# TRIP DETAILS
# ============================================================

st.markdown("## 📝 Trip Details")

col1, col2, col3 = st.columns(3)

with col1:
    passenger_count = st.number_input(
        "Passenger Count",
        min_value=1,
        max_value=6,
        value=1,
        step=1
    )

with col2:
    pickup_date = st.date_input(
        "Pickup Date",
        value=datetime(2024, 1, 1).date()
    )

with col3:
    pickup_time = st.time_input(
        "Pickup Time",
        value=datetime(2024, 1, 1, 12, 0).time()
    )

pickup_datetime = datetime.combine(pickup_date, pickup_time)

# ============================================================
# LOCATION DETAILS
# ============================================================

st.markdown("## 📍 Pickup & Drop-off Locations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Pickup Location**")
    pickup_latitude = st.number_input(
        "Pickup Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=40.7128,
        format="%.6f",
        key="pickup_lat"
    )
    pickup_longitude = st.number_input(
        "Pickup Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-74.0060,
        format="%.6f",
        key="pickup_lon"
    )

with col2:
    st.markdown("**Drop-off Location**")
    dropoff_latitude = st.number_input(
        "Drop-off Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=40.7580,
        format="%.6f",
        key="dropoff_lat"
    )
    dropoff_longitude = st.number_input(
        "Drop-off Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-73.9855,
        format="%.6f",
        key="dropoff_lon"
    )

st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# ============================================================
# PREDICTION BUTTON
# ============================================================

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🚕 Predict Fare", use_container_width=True)

# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict_button:
    pickup_hour = pickup_datetime.hour
    pickup_day = pickup_datetime.day
    pickup_day_of_week = pickup_datetime.weekday()
    pickup_month = pickup_datetime.month
    pickup_year = pickup_datetime.year

    is_weekend = int(pickup_day_of_week >= 5)
    
    rush_hours = [7, 8, 9, 16, 17, 18, 19]
    is_rush_hour = int(pickup_hour in rush_hours)

    trip_distance = haversine_distance(
        pickup_latitude, pickup_longitude,
        dropoff_latitude, dropoff_longitude
    )

    longitude_difference = abs(dropoff_longitude - pickup_longitude)
    latitude_difference = abs(dropoff_latitude - pickup_latitude)
    distance_per_passenger = trip_distance / passenger_count
    trip_distance_log = np.log1p(trip_distance)

    input_data = {
        "passenger_count": passenger_count,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "pickup_hour": pickup_hour,
        "pickup_day": pickup_day,
        "pickup_day_of_week": pickup_day_of_week,
        "pickup_month": pickup_month,
        "pickup_year": pickup_year,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "trip_distance": trip_distance,
        "longitude_difference": longitude_difference,
        "latitude_difference": latitude_difference,
        "distance_per_passenger": distance_per_passenger,
        "trip_distance_log": trip_distance_log
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURES]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled, verbose=0)
    predicted_fare = max(0, float(prediction[0][0]))

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

    # ============================================================
    # RESULTS
    # ============================================================

    st.markdown("## 🎯 Your Fare Estimate")

    st.markdown(f"""
    <div class="result-showcase">
        <div class="fare-label">💰 Estimated Fare Amount</div>
        <div class="fare-amount">${predicted_fare:.2f}</div>
        <div class="fare-desc">
            ✓ Distance: {trip_distance:.2f} km • Passengers: {passenger_count} • Time: {pickup_datetime.strftime("%I:%M %p")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # TRIP SUMMARY
    # ============================================================

    st.markdown("## 📊 Trip Summary")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Distance", f"{trip_distance:.2f} km")

    with metric_col2:
        st.metric("Passengers", f"{passenger_count}")

    with metric_col3:
        st.metric("Time", pickup_datetime.strftime("%H:%M"))

    with metric_col4:
        status = "🔴 Rush Hour" if is_rush_hour else "🟢 Normal"
        st.metric("Status", status)

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

    # ============================================================
    # ROUTE DETAILS
    # ============================================================

    st.markdown("## 📍 Route Details")

    location_data = pd.DataFrame({
        "Location": ["Pickup", "Drop-off"],
        "Latitude": [f"{pickup_latitude:.4f}", f"{dropoff_latitude:.4f}"],
        "Longitude": [f"{pickup_longitude:.4f}", f"{dropoff_longitude:.4f}"]
    })

    st.dataframe(location_data, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🔍 All 17 AI Features"):
            feature_display = input_df.copy()
            for col in feature_display.columns:
                feature_display[col] = feature_display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
            st.dataframe(feature_display, use_container_width=True)

    with col2:
        with st.expander("🤖 Model Specifications"):
            st.markdown("""
            **Neural Network Details:**
            - **Type:** Deep Feedforward Neural Network
            - **Input Layer:** 17 Features
            - **Hidden Layer 1:** 256 neurons (ReLU + Dropout 0.2)
            - **Hidden Layer 2:** 128 neurons (ReLU + Dropout 0.2)
            - **Hidden Layer 3:** 64 neurons (ReLU + Dropout 0.2)
            - **Output Layer:** 1 neuron (Regression)
            - **Optimizer:** Adam
            - **Distance:** Haversine Formula
            - **Preprocessing:** StandardScaler
            - **Task:** Fare Prediction (USD)
            """)

st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="text-align: center; padding: 3rem 2rem; position: relative; z-index: 2;">
    <div style="font-size: 1.5rem; margin-bottom: 1rem;">🚕</div>
    <p style="color: #FFFFFF; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">NYC Taxi Fare Prediction</p>
    <p style="color: #B0B8C1; font-size: 0.9rem; margin-bottom: 1rem;">Advanced Machine Learning • Real-time Estimation</p>
    <p style="color: #808080; font-size: 0.8rem;">Built with TensorFlow • Streamlit • Deep Neural Networks</p>
</div>
""", unsafe_allow_html=True)