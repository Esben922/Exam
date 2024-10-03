import streamlit as st

pg = st.navigation([st.Page("EDA.py"), st.Page("Loan Amount Prediction.py"), st.Page("Term Prediction.py")])
pg.run()