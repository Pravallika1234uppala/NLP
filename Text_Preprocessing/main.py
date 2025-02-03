import streamlit as st
from gensim.models import KeyedVectors
import gdown
import os

#Loading the traioned model:
def load_word2vec():
    model_url="https://drive.google.com/uc?id=1NXe6o5kEluV86RWO7nEV5WUQnS6u6_EV"

    if not os.path.exists(model_file):
        gdown.download(model_url, quiet=False)

    # Load the model
    return KeyedVectors.load_word2vec_format(model_url, binary=True)

model = load_word2vec()

#starting streamlit app-
#Giving title to app
st.title('Word and Similar Word Prediction')

#word Input:

input_word = st.text_input("Enter a word:")

if input_word:
    try:
        vec = model[input_word]
        prediction = model.most_similar([vec], topn=5)

        similar_words = [wrd for wrd, _ in prediction]

        st.write("Most similar words:")
        for similar_word in similar_words:
            st.write(f"{similar_word}")
    
    except KeyError:
        st.write("Word not found in the vocabulary. Try another word!")
