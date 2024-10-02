import streamlit as st

pg = st.navigation([st.Page("EDA.py"), st.Page("SML Prediction.py")])
pg.run()