import streamlit as st

pg = st.navigation([st.Page("EDA.py"), st.Page("Loan Amount Prediction.py")])
pg.run()


# Footer
st.markdown("---")
st.markdown("Developed by Camilla Louise Jensen, Esben Graahede, Imran Talukder, Piyal Dey, & Samil Demiroglu")