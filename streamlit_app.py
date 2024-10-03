import streamlit as st

pg = st.navigation([st.Page("EDA.py"), st.Page("Loan Amount Prediction.py")])
pg.run()