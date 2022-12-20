from utils.libraries import st, pd
from utils.functions import spacy_process, load_model

def show_predict_page():

    st.title("Predict If Job is Real or Fake")

    text = st.text_area('**Enter Job Description**')

    ok = st.button("Predict")

    if ok:

        st.write("**Input Text**")
        st.write(text)

        text = spacy_process(text)
        st.write("**After Text-Preprocessing**")
        st.write(text)

        data = {
            'text': [text]
        }

        df = pd.DataFrame(data)

        data = load_model()
        model = data["model"]
        vectorizer = data["vectorizer"]

        x = vectorizer.transform(df.loc[:,'text'])
        temp = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

        prediction = model.predict(temp)

        if prediction[0] == 1:
            st.markdown(unsafe_allow_html=True,
                    body="<span style='color:red; font-size: 50px'><strong><h4>Job is Fake! :slightly_frowning_face:</h4></strong></span>")
        elif prediction[0] == 0:
            st.markdown(unsafe_allow_html=True,
                    body="<span style='color:green; font-size: 50px'><strong><h3>Job is Real! :smile: </h3></strong></span>")
    


