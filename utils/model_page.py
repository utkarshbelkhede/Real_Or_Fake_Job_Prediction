from utils.libraries import *
from utils.functions import train_model, evaluate, trainer


def compare_model_page():

    button = False

    st.title("Model Page")

    df = pd.read_csv(r'datasets\clean_df.csv')

    st.write("#### 1. Vectorizer Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        gram = st.selectbox("**Select Grams**", ("Mono-Gram", "Bi-Gram", "Tri-Gram"))
        
        if gram == "Mono-Gram":
            gram = (1,1)
        elif gram == "Bi-Gram":
            gram = (2,2)
        elif gram == "Tri-Gram":
            gram = (3,3)

    with col2:
        no_features = st.slider('**Select Max-Features**', 1, 1000, 100)

    with col3:
        vec = st.selectbox("**Select Vectorizer**", ("Count", "TF-IDF"))

    if vec == "Count":
        vectorizer = CountVectorizer(ngram_range=gram, max_features = no_features)
    elif vec == "TF-IDF":
        vectorizer = TfidfVectorizer(ngram_range=gram, max_features = no_features)

    model = st.selectbox("**Select Model**", ("Logistic Regression","Random Forest","Support Vector Machine"))

    st.write("#### 2. Data Configuration")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider('**Select Test Size**', 10, 100, 30)
        test_size = test_size/100

    with col2:
        over_sample = st.selectbox('**Do Over-Sampling**', ['Yes', 'No'])
        if over_sample == 'Yes':
            over_sample = True
        elif over_sample == 'No':
            over_sample = False

    st.write("#### 3. Model Configuration")

    if model == "Logistic Regression":

        col1, col2 = st.columns(2)

        with col1:
            penalty = st.selectbox("**Select Penalty**", ("l1","l2","elasticnet"))
            random_state = st.slider('**Select Random State**', 1, 1000, 42)
        with col2:
            solver = st.selectbox("**Select Solver**", ("liblinear","newton-cg", "newton-cholesky", "sag", "saga"))
            n_jobs = st.slider('**Select N-Jobs**', 1, 1000, 42)

        model = LogisticRegression(
            penalty=penalty,
            solver=solver,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        train = st.button("Train")

        if train:
            st.write("#### 4. Model Evaluation")
            trainer(df, test_size, over_sample, vectorizer, model)
            button = st.button('Save Logistic Regression as Pickle')

    elif model == "Random Forest":
        col1, col2 = st.columns(2)

        with col1:
            criterion = st.selectbox("**Select Criterion**", ("gini","entropy","elasticnet"))
            n_estimators = st.slider('**Select N-Estimatorse**', 1, 1000, 100)
            n_jobs = st.slider('**Select N-Jobs**', 1, 1000, 10)
        with col2:
            max_features = st.selectbox("**Select Max-Features**", ("sqrt","log2"))
            max_depth = st.slider('**Select Max-Depth**', 1, 50, 10)
            random_state = st.slider('**Select Random-State**', 1, 1000, 42)

        model = RandomForestClassifier(
            criterion=criterion,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )

        train = st.button("Train")

        if train:
            st.write("#### 4. Model Evaluation")
            trainer(df, test_size, over_sample, vectorizer, model)
            button = st.button('Save Random Forest as Pickle')

    elif model == "Support Vector Machine":
        col1, col2 = st.columns(2)

        with col1:
            kernel = st.selectbox("**Select Kernel**", ("linear","poly","rbf", "sigmoid"))

        with col2:
            random_state = st.slider('**Select Random State**', 1, 1000, 42)

        model = SVC(
            kernel=kernel,
            random_state=random_state,
            probability=True
        )

        train = st.button("Train")

        if train:
            st.write("#### 4. Model Evaluation")
            trainer(df, test_size, over_sample, vectorizer, model)
            button = st.button('Save Support Vector Machine as Pickle')

    if button:
        data = {"model": model}
        with open(r'pickle\app_model.pkl', 'wb') as file:
            pickle.dump(data, file)
