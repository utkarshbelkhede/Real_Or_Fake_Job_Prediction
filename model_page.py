import streamlit as st
import math
import pickle
import re
import string
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix

from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from explore_page import split, text_fraud_df

# Create our list of punctuation marks
punctuation = string.punctuation

# Create our list of stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()


# Create our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our taken object, which is used to create documents with linguistic annotations
    my_tokens = parser(sentence)

    # Legitimating each taken and converting each taken into lowercase
    my_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in my_tokens]

    my_tokens = [word for word in my_tokens if word not in stop_words and word not in punctuation]

    return my_tokens


# Custom transformer using spaCy
class predictors(TransformerMixin):

    def transform(self, X, **transform_parmas):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into Lowercase
    return text.strip().lower()


@st.cache(ttl=24 * 60 * 60)
def term_freq(temp_df):
    temp_df['text'] = temp_df['text'].apply(clean_text)

    cv = TfidfVectorizer(max_features=100)
    x = cv.fit_transform(temp_df['text'])

    df = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())

    temp_df.drop(['text'], axis=1, inplace=True)

    return pd.concat([df, temp_df], axis=1)


def compare_model_page():

    st.title("Let's Compare Models")

    jobs = pd.read_csv("/home/utkarsh/PycharmProjects/Real_Or_Fake_Job/Datasets/fake_job_postings.csv")

    columns = ['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']

    for col in columns:
        del jobs[col]

    jobs.fillna('', inplace=True)

    jobs['country'] = jobs.location.apply(split)

    jobs = text_fraud_df(jobs)

    st.write("""
    ### Converting Sentences into vectors\n ### Using Term-Frequency and Inverse-Document-Frequency
    """)
    main_df = term_freq(jobs)

    st.dataframe(main_df.head())

    shape = main_df.shape
    st.write("There are", shape[0], "rows and ", shape[1], "columns in the Dataset.")

    Y = main_df.iloc[:, -1]
    X = main_df.iloc[:, :-1]

    st.write("""
    #### Test Data Percentage
    """)
    test_data = st.slider(' ', 10, 90, value=30)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_data / 100)

    st.write("""
        #### Choose Model
        """)
    choice = st.selectbox(' ', ['Random Forest', 'Support Machine Vector'])

    if choice == "Random Forest":
        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            n_estimators = math.ceil(st.number_input('n_estimators', 0, 150, value=100))
        with col_2:
            n_jobs = math.ceil(st.number_input('n_jobs', 0, 10, value=3))
        with col_3:
            max_depth = math.ceil(st.number_input('max_depth', 0, 10, value=0))

        if max_depth == 0:
            max_depth = None

        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            oob_score = st.checkbox('oob_score')

        with col_2:
            criterion = st.radio('criterion', ['gini', 'entropy'])

        with col_3:
            max_features = st.radio('max_features', ['auto', 'sqrt', 'log2'])

        rfc = RandomForestClassifier(n_jobs=n_jobs, oob_score=oob_score, n_estimators=n_estimators, criterion=criterion,
                                     max_depth=max_depth, max_features=max_features)
        rfc.fit(X_train, y_train)

        rfc_pred = rfc.predict(X_test)
        rfc_score = accuracy_score(y_test, rfc_pred)

        st.write(""" #### Classification Report """)
        st.text(classification_report(y_test, rfc_pred))

        st.write(""" #### Accuracy Score is {0:.2f} %""".format(rfc_score * 100))

        button = st.button('Save Random Forest as Pickle')

        if button:
            data = {"model": rfc}
            with open('/home/utkarsh/PycharmProjects/Real_Or_Fake_Job/Pickle/random_forest.pkl', 'wb') as file:
                pickle.dump(data, file)


    elif choice == 'Support Machine Vector':

        kernel = st.radio('kernel', ['rbf','linear', 'poly','sigmoid'])

        svc = SVC(kernel=kernel, random_state=0)
        svc.fit(X_train, y_train)

        svc_pred = svc.predict(X_test)
        svc_score = accuracy_score(y_test, svc_pred)

        st.write(""" #### Classification Report """)
        st.text(classification_report(y_test, svc_pred))

        st.write(""" #### Accuracy Score is {0:.2f} %""".format(svc_score * 100))

        button = st.button('Save Support Vector Machine as Pickle')

        if button:
            data = {"model": svc}
            with open('/home/utkarsh/PycharmProjects/Real_Or_Fake_Job/Pickle/support_vector.pkl', 'wb') as file:
                pickle.dump(data, file)
