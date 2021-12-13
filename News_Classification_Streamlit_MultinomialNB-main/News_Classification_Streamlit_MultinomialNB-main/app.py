import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
#NLP
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#load model
modell = pickle.load(open('final_model', 'rb'))
tfidf = pickle.load(open('tfidf', 'rb'))


#function for preprocessing
def preprocess_text(text):
    # remove all punctuation
    text = re.sub(r'[^\w\d\s]', ' ', text)
    # collapse all white spaces
    text = re.sub(r'\s+', ' ', text)
    # convert to lower case
    text = re.sub(r'^\s+|\s+?$', '', text.lower())
    #taking only alphabetic words
    text = re.sub('[^a-zA-Z]',' ',text)
    # removing only single letters in the
    text= ' '.join( [w for w in text.split() if len(w)>1])
    # remove stop words and perform stemming
    stop_words = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    return ' '.join(
        lemmatizer.lemmatize(term)
        for term in text.split()
        if term not in set(stop_words)
    )

def input_predict(text):
    # preprocess the text
    text = preprocess_text(text)
    # convert text to a list
    yh = [text]
    # transform the input
    inputpredict = tfidf.transform(yh)
    # predict the user input text
    y_predict_userinput = modell.predict(inputpredict)
    output = int(y_predict_userinput)

    return output

def main():

    st.title("Text Classifier")
    #activities = ["Summarize", "Classify"]
    #choice = st.sidebar.selectbox("Select Activity", activities)

    #if choice == "Summarize":
    st.subheader("Classify Text with NLP Model")
    raw_text = st.text_area("Enter Your Text", "Type Here or Paste")

    #summary_choice = st.selectbox("Summary Choice", ["Gensim", "Sumy Lex Rank"])
    if st.button("Classify"):
        #processed_text = preprocess_text(raw_text)
        predict_input = input_predict(raw_text)

        if predict_input == 0:
            category = "Business"
        elif predict_input == 1:
            category = "Entertainment"
        elif predict_input == 2:
            category = "Politics"
        elif predict_input == 3:
            category = "Sports"
        elif predict_input == 4:
            category = "Technology"
        else:
            category = "Error"

        st.write(category)

    else:
        st.write("Provide Text to classify")

    return None
if __name__ == "__main__":
    main()
