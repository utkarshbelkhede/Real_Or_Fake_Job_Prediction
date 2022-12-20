from utils.libraries import st, pd, plt, sns

def show_explore_page():
    st.title("Explore Jobs Dataset")
    
    # Loading Dataset into DataFrame
    st.write("#### 1. About Dataset")
    df = pd.read_csv(r"datasets\fake_job_postings.csv")
    st.dataframe(df.head())

    rows = df.shape[0]
    cols = df.shape[1]

    st.write("This dataset has", rows, "rows and ",cols, "columns.")

    st.write("#### 2. Exploratory Data Analysis")

    st.write("##### 2.1 Missing Values")

    fig = sns.set(rc={'figure.figsize': (8, 5)})
    fig, ax = plt.subplots()
    plt.title("Heat Map for Missing Values")
    sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
    st.pyplot(fig)

    # Filling Na with Blank Spaces
    df.fillna('', inplace=True)

    st.write("##### 2.2 Comparing Number of Fraudlent and Non-Fraudlent Job Posting")

    fig = sns.set(rc={'figure.figsize': (10, 3)})
    fig, ax = plt.subplots()
    plt.title("Number of Fradulent Vs Non-Fraudlent Jobs")
    sns.countplot(y='fraudulent', data=df)
    st.pyplot(fig)

    not_fraudulent = df.groupby('fraudulent')['fraudulent'].count()[0]
    fraudulent = df.groupby('fraudulent')['fraudulent'].count()[1]

    st.write(not_fraudulent, "jobs are NOT Fraudulent and ", fraudulent, " jobs are Fraudulent.")

    st.write("##### 2.3 Experiencewise Count")

    exp = dict(df.required_experience.value_counts())
    del exp['']

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    plt.bar(exp.keys(),exp.values())
    plt.title('No. of Jobs with Experience')
    plt.xlabel('Experience')
    plt.ylabel('No. of jobs')
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.write("##### 2.4 Countrywise Job Count")

    # First Spliting location Column to extract Country Code
    def split(location):
        l = location.split(',')
        return l[0]

    df['country'] = df.location.apply(split)

    countr = dict(df.country.value_counts()[:14])
    del countr['']

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    plt.title('Country-wise Job Posting')
    plt.bar(countr.keys(), countr.values())
    plt.ylabel('No. of jobs')
    plt.xlabel('Countries')
    st.pyplot(fig)

    st.write("##### 2.5 Education Job Count")

    edu = dict(df.required_education.value_counts()[:7])
    del edu['']

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    plt.title('Job Posting based on Education')
    plt.bar(edu.keys(), edu.values())
    plt.ylabel('No. of jobs')
    plt.xlabel('Education')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write("##### 2.6 Top 10 Titles of Jobs Posted which were NOT fraudulent")

    dic = dict(df[df.fraudulent==0].title.value_counts()[:10])
    dic_df = pd.DataFrame.from_dict(dic, orient ='index')
    dic_df.columns = ["Number of Jobs"]
    st.dataframe(dic_df)

    st.write("##### 2.7 Top 10 Titles of Jobs Posted which were fraudulent")

    dic = dict(df[df.fraudulent==1].title.value_counts()[:10])
    dic_df = pd.DataFrame.from_dict(dic, orient ='index')
    dic_df.columns = ["Number of Jobs"]
    st.dataframe(dic_df)
