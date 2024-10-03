import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
from streamlit_shap import st_shap


st.title("Prediction using Supervised Machine Learning")

# Load model and preprocessing objects
@st.cache_resource
def load_model_objects():
    model_xgb = joblib.load('model_xgb1.joblib')
    scaler = joblib.load('scaler1.joblib')
    ohe = joblib.load('ohe1.joblib')
    return model_xgb, scaler, ohe

model_xgb, scaler, ohe = load_model_objects()

# Create SHAP explainer
explainer = shap.TreeExplainer(model_xgb)

# App description
with st.expander("What's this app?"):
    st.markdown("""
    This app is a KIVA loan term predicter!
    """)

st.subheader('Describe your situation')

# User inputs
col1, col2 = st.columns(2)

with col1:
    sector = st.selectbox('sector', options=ohe.categories_[0])
    country = st.selectbox('country', options=ohe.categories_[1])
    borrower_genders = st.selectbox('borrower_genders', options=ohe.categories_[2])
    repayment_interval = st.selectbox('repayment_interval', options=ohe.categories_[3])

with col2:
    funded_amount = st.number_input('funded_amount', min_value=0, max_value=10000, value=1)
    loan_amount = st.number_input('loan_amount', min_value=0, max_value=10000, value=1)
    term_in_months = st.number_input('term_in_months', min_value=0, max_value=100, value=1)
    lender_count = st.number_input('lender_count', min_value=0, max_value=100, value=1)

# Prediction button
if st.button('Predict Funding Time!'):
    # Prepare categorical features
    cat_features = pd.DataFrame({'sector': [sector],'country': [country],'borrower_genders': [borrower_genders],'repayment_interval': [repayment_interval]})
    cat_encoded = pd.DataFrame(ohe.transform(cat_features).toarray(), columns=ohe.get_feature_names_out())

    # Prepare numerical features
    num_features = pd.DataFrame({
        'funded_amount': [funded_amount],
        'loan_amount': [loan_amount],
        'term_in_months': [term_in_months],
        'lender_count': [lender_count],
    })
    num_scaled = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
    
    # Combine features
    features = pd.concat([num_scaled, cat_encoded], axis=1)
    
    # Make prediction
    predicted_months = model_xgb.predict(features)[0]
    
    # Display prediction
    st.metric(label="Predicted Loan:", value=f'{round(predicted_months)} months')
    
    # Calculate and display price range
    lower_range = max(0, round(predicted_months - 7))
    upper_range = round(predicted_months + 7)
    st.write(f"Potential Variation: {lower_range} - {upper_range} months")
    
    # SHAP explanation
    st.subheader('Price Factors Explained 🤖')
    shap_values = explainer.shap_values(features)
    st_shap(shap.force_plot(explainer.expected_value, shap_values, features), height=400, width=600)
    
    st.markdown("""
    This plot shows how each feature contributes to the predicted price:
    - Blue bars push the months become lower
    - Red bars push the months become higher
    - The length of each bar indicates the strength of the feature's impact
    """)