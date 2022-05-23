import streamlit as st
import pandas as pd
import pickle

from explore_page import split, text_fraud_df
from model_page import term_freq


def load_model(model):
    if model == "Random Forest":
        with open('/home/utkarsh/PycharmProjects/Real_Or_Fake_Job/Pickle/random_forest.pkl', 'rb') as file:
            data = pickle.load(file)
    elif model == "Support Machine Vector":
        with open('/home/utkarsh/PycharmProjects/Real_Or_Fake_Job/Pickle/support_vector.pkl', 'rb') as file:
            data = pickle.load(file)
    return data


def show_predict_page():
    st.title("Let's Predict Job is Fake!")

    st.write("""### We need some information to predict""")

    file = st.file_uploader('Upload a CSV')

    if file is not None:
        test = pd.read_csv(file)
        test_copy = test.copy()

        st.dataframe(test.head())

        algo = st.radio('Algo', ['Random Forest', 'Support Machine Vector'])

        data = load_model(algo)

        model = data["model"]

        ok = st.button("Predict")
        if ok:

            jobs = pd.read_csv("/home/utkarsh/PycharmProjects/Real_Or_Fake_Job/Datasets/fake_job_postings.csv")

            columns = ['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range',
                       'employment_type']

            for col in columns:
                del jobs[col]

            jobs.fillna('', inplace=True)

            jobs['country'] = jobs.location.apply(split)

            jobs = text_fraud_df(jobs)
            _check = jobs.copy()

            test = text_fraud_df(test)

            new = _check.append(test, ignore_index=True)

            main_test = term_freq(new)
            final_test = main_test.tail()[-2:].drop('fraudulent', axis=1)

            st.subheader("Prediction")

            test_copy["Result"] = "Real"
            for i in range(model.predict(final_test).size):
                if model.predict(final_test)[i] == 1:
                    test_copy.iloc[i, -1] = "Fake"

            st.dataframe(test_copy)

