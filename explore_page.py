import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


def split(location):
    l = location.split(',')
    return l[0]


def text_fraud_df(jobs):
    jobs.fillna('', inplace=True)
    jobs['text'] = jobs['title'] + ' ' + jobs['company_profile'] + ' ' + jobs['description'] + jobs[
        'requirements'] + ' ' + jobs['benefits']

    for col in jobs.columns:
        if col not in ['text', 'fraudulent']:
            del jobs[col]

    return jobs


def show_explore_page():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    jobs = pd.read_csv("/home/utkarsh/PycharmProjects/Real_Or_Fake_Job/Datasets/fake_job_postings.csv")

    st.title("Explore Jobs Dataset")

    st.write(
        """
    ### Jobs
    """
    )
    st.dataframe(jobs.head())

    shape = jobs.shape
    st.write("There are", shape[0], "rows and ", shape[1], "columns in the Dataset.")

    null_features = [[features, jobs[features].isnull().sum()] for features in jobs.columns if
                     jobs[features].isnull().sum() > 0]

    null_features = pd.DataFrame(null_features, columns=['Feature', 'No. of Null Values'])

    st.write(
        """
    ### Null Values in Dataset
    """
    )
    st.dataframe(null_features)

    st.write("""
        ### Heatmap of Missing Values
        """)

    sns.heatmap(jobs.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    fig = sns.set(rc={'figure.figsize': (14, 10)})
    st.pyplot(fig)

    st.write("""
            ### Dropping unnecessary Columns
            """)

    columns = ['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']

    st.write(columns)
    for col in columns:
        del jobs[col]

    st.dataframe(jobs.head())

    jobs.fillna('', inplace=True)

    st.write("""
    ### Comparing Number of Fraudulent and Non-Fraudulent Job Posting
    """)

    fig = plt.figure(figsize=(15, 5))
    sns.countplot(y='fraudulent', data=jobs)
    st.pyplot(fig)

    not_fraudulent = jobs.groupby('fraudulent')['fraudulent'].count()[0]
    fraudulent = jobs.groupby('fraudulent')['fraudulent'].count()[1]

    st.write(not_fraudulent, "jobs are NOT Fraudulent", fraudulent, "jobs are Fraudulent")

    st.write("""
    ### Bar Plot
    #### Experience Vs No. of Jobs
    """)

    exp = dict(jobs.required_experience.value_counts())
    del exp['']

    fig = plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    plt.bar(exp.keys(), exp.values())
    plt.title('No. of Jobs with Experience', size=20)
    plt.xlabel('Experience', size=10)
    plt.ylabel('No. of jobs', size=10)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.write("""
    ### Experiencewise Count
    """)

    exp_df = pd.DataFrame.from_dict(exp, orient='index')
    exp_df.columns = ['Count']
    st.dataframe(exp_df)

    st.write("""
    ### Bar Plot
    #### Top 14 Country Vs No. of Jobs
    """)

    jobs['country'] = jobs.location.apply(split)

    countr = dict(jobs.country.value_counts()[:14])
    del countr['']

    fig = plt.figure(figsize=(8, 6))
    plt.title('Country-wise Job Posting', size=20)
    plt.bar(countr.keys(), countr.values())
    plt.ylabel('No. of jobs', size=10)
    plt.xlabel('Countries', size=10)
    st.pyplot(fig)

    countr_df = pd.DataFrame.from_dict(countr, orient='index')
    countr_df.columns = ['Count']
    st.dataframe(countr_df.T)

    st.write("""
        ### Bar Plot
        #### Education Vs No. of Jobs
        """)

    edu = dict(jobs.required_education.value_counts()[:7])
    del edu['']

    fig = plt.figure(figsize=(15, 6))
    plt.title('Job Posting based on Education', size=20)
    plt.bar(edu.keys(), edu.values())
    plt.ylabel('No. of jobs', size=10)
    plt.xlabel('Education', size=10)
    st.pyplot(fig)

    edu_df = pd.DataFrame.from_dict(edu, orient='index')
    edu_df.columns = ['Count']
    st.dataframe(edu_df)

    st.write("""
    ### Top 10 Titles of Jobs Posted which were NOT fraudulent
    """)

    st.write(jobs[jobs.fraudulent == 0].title.value_counts()[:10])

    st.write("""
    ### Top 10 Titles of Jobs Posted which were fraudulent
    """)

    st.write(jobs[jobs.fraudulent == 1].title.value_counts()[:10])

    st.write("""
        ### Combining Text Features into One Column
        """)
    jobs = text_fraud_df(jobs)

    st.dataframe(jobs.head())

    st.write("""
        ### Creating a WordCloud based on Fraudulent Jobs
        """)

    fraud_jobs_text = jobs[jobs.fraudulent == 1].text

    STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
    plt.figure(figsize=(16, 14))
    wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800, stopwords=STOPWORDS,
                   background_color="white").generate(
        str(" ".join(fraud_jobs_text)))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()
