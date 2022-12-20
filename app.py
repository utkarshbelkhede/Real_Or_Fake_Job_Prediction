from utils.libraries import st
from utils.explore_page import show_explore_page
from utils.model_page import compare_model_page
from utils.predict_page import show_predict_page


page = st.sidebar.selectbox("Explore Or Predict Or Else", ("Understanding the Data","Compare Models","Predict"))

if page == "Understanding the Data":
    show_explore_page()
elif page == "Compare Models":
    compare_model_page()
elif page == "Predict":
    show_predict_page()