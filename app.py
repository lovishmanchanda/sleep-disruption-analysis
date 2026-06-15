import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
@st.cache_resource
def load_models():
    clf_model = joblib.load('models/rf_model.joblib')
    clf_scaler = joblib.load('models/scaler.joblib')
    clf_features = joblib.load('models/features.joblib')
    
    reg_model = joblib.load('models/regression_model.joblib')
    reg_scaler = joblib.load('models/regression_scaler.joblib')
    reg_features = joblib.load('models/regression_features.joblib')
    
    return clf_model, clf_scaler, clf_features, reg_model, reg_scaler, reg_features

st.set_page_config(page_title="Sleep Disruption & Attention Fragmentation", layout="centered")

clf_model, clf_scaler, clf_features, reg_model, reg_scaler, reg_features = load_models()

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }
    
    h1 {
        font-weight: 800 !important;
        text-align: center;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        font-weight: 600 !important;
        padding-bottom: 8px;
        margin-top: 30px;
    }
    
    /* Style the form container to look like a clean card using native theme colors */
    div[data-testid="stForm"] {
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        padding: 32px;
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Beautiful Gradient Button */
    div.stButton > button:first-child, div.stFormSubmitButton > button:first-child {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white !important;
        border: none;
        width: 100%;
        padding: 14px 24px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        transition: all 0.2s ease;
    }
    
    div.stButton > button:first-child:hover, div.stFormSubmitButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5);
    }
    
    /* Large Metric Text */
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #4F46E5 !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Sleep Disruption and Attention Fragmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.15rem; margin-bottom: 2rem; opacity: 0.8;'>A machine learning approach to understanding how your digital habits and lifestyle impact your cognitive focus and rest.</p>", unsafe_allow_html=True)

with st.form("prediction_form"):
    st.markdown("### Digital Habits")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Total Daily Screen Time**")
        sc1, sc2 = st.columns(2)
        with sc1:
            screen_time_hrs = st.number_input("Hours", 0, 24, 5, key="st_hrs")
        with sc2:
            screen_time_mins = st.number_input("Minutes", 0, 59, 0, key="st_mins")
    
    with col2:
        st.markdown("**Phone Usage Before Sleep**")
        ph1, ph2 = st.columns(2)
        with ph1:
            phone_sleep_hrs = st.number_input("Hours", 0, 5, 1, key="ph_hrs")
        with ph2:
            phone_sleep_mins = st.number_input("Minutes", 0, 59, 0, key="ph_mins")
            
    st.markdown("### Lifestyle & Health")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Average Sleep Duration**")
        sl1, sl2 = st.columns(2)
        with sl1:
            sleep_hrs = st.number_input("Hours", 0, 14, 7, key="sl_hrs")
        with sl2:
            sleep_mins = st.number_input("Minutes", 0, 59, 0, key="sl_mins")
    with col4:
        st.markdown("**Daily Physical Activity**")
        ac1, ac2 = st.columns(2)
        with ac1:
            activity_hrs = st.number_input("Hours", 0, 5, 0, key="act_hrs")
        with ac2:
            activity_mins = st.number_input("Minutes", 0, 59, 45, key="act_mins")
            
    caffeine = st.number_input("Daily Caffeine Intake (Cups)", 0, 15, 2)
    
    st.markdown("### Wellbeing & Demographics")
    col5, col6 = st.columns(2)
    with col5:
        stress = st.slider("Stress & Fatigue Level (1 = Low, 10 = High)", 1.0, 10.0, 5.0)
        digital_wellbeing_rating = st.slider("Digital Life Balance (1 = Poor, 10 = Excellent)", 1, 10, 6)
    with col6:
        age = st.number_input("Age", 18, 90, 30)
        occupations = ['Designer', 'Doctor', 'Freelancer', 'Manager', 'Researcher', 'Software Engineer', 'Student', 'Teacher', 'Other']
        selected_occupation = st.selectbox("Occupation", occupations)
        
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Generate Predictive Report")

if submitted:
    with st.spinner('Analyzing your profile with ML Models...'):
        total_screen_time_mins = (screen_time_hrs * 60) + screen_time_mins
        total_phone_sleep_mins = (phone_sleep_hrs * 60) + phone_sleep_mins
        total_sleep_duration_mins = (sleep_hrs * 60) + sleep_mins
        total_activity_mins = (activity_hrs * 60) + activity_mins
        digital_wellbeing_score = digital_wellbeing_rating * 10
        
        master_data = {
            'daily_screen_time_minutes': [total_screen_time_mins],
            'phone_usage_before_sleep_minutes': [total_phone_sleep_mins],
            'sleep_duration_minutes': [total_sleep_duration_mins],
            'stress_fatigue_index': [stress],
            'digital_wellbeing_score': [digital_wellbeing_score],
            'caffeine_intake_cups': [caffeine],
            'physical_activity_minutes': [total_activity_mins],
            'age': [age]
        }
        
        all_features = set(clf_features + reg_features)
        for occ in all_features:
            if occ.startswith('occupation_'):
                occ_name = occ.split('_', 1)[1].replace('_', ' ').title()
                master_data[occ] = [1 if occ_name == selected_occupation else 0]
                
        master_df = pd.DataFrame(master_data)
        
        # Regression Prediction
        reg_input = master_df[reg_features]
        reg_scaled = pd.DataFrame(reg_scaler.transform(reg_input), columns=reg_features)
        sleep_quality_pred = max(0, min(10, reg_model.predict(reg_scaled)[0]))
        
        # Classification Prediction
        clf_input = master_df[clf_features]
        clf_scaled = pd.DataFrame(clf_scaler.transform(clf_input), columns=clf_features)
        attention_pred = clf_model.predict(clf_scaled)[0]
        class_names = {0: 'Low', 1: 'Moderate', 2: 'High'}
        attention_label = class_names.get(attention_pred, str(attention_pred))

        st.markdown("<h2 style='text-align: center; margin-top: 40px;'>Your Diagnostic Report</h2>", unsafe_allow_html=True)
        
        rc1, rc2 = st.columns(2)
        
        with rc1:
            st.metric(label="Predicted Sleep Quality", value=f"{sleep_quality_pred:.1f} / 10")
            if sleep_quality_pred >= 8.0:
                st.success("Excellent sleep quality expected. Keep up your healthy routines.")
            elif sleep_quality_pred >= 6.0:
                st.warning("Moderate sleep quality. Consider reducing screen time or caffeine.")
            else:
                st.error("Poor sleep quality expected. High risk of sleep disruption.")
                
        with rc2:
            st.metric(label="Attention Fragmentation", value=f"{attention_label}")
            if attention_label in ['Low', '0']:
                st.success("You likely maintain strong focus and are highly productive.")
            elif attention_label in ['Moderate', '1']:
                st.warning("You may experience occasional distractions throughout the day.")
            else:
                st.error("High fragmentation detected. Try digital detoxing to regain focus.")

# --- Footer Section ---
st.markdown("---")
footer_html = """
<div style='text-align: center; color: #64748B; font-size: 0.95rem; margin-top: 20px;'>
    <p><b>Project by Lovish Manchanda</b></p>
    <p>
        <a href='mailto:lovishmanchanda.work@gmail.com' style='color: #4F46E5; text-decoration: none; font-weight: 600;'>Contact via Email</a> | 
        <a href='https://github.com/lovishmanchanda' target='_blank' style='color: #4F46E5; text-decoration: none; font-weight: 600;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16" style="vertical-align: text-bottom; margin-right: 4px;">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>GitHub
        </a>
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
