import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from model_page import compare_model_page


page = st.sidebar.selectbox("Explore Or Predict Or Else", ("Understanding the Data","Compare Models","Predict"))

if page == "Understanding the Data":
    show_explore_page()
elif page == "Compare Models":
    compare_model_page()
elif page == "Predict":
    show_predict_page()
