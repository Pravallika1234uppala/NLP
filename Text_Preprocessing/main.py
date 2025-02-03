import streamlit as st
import gensim
import gensim.downloader as api
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')


model = api.load('glove-wiki-gigaword-100')

#starting streamlit app-
#Giving title to app
st.title('Word Predictions')

#word Input:

input_word = st.text_input("Enter a word:")

def get_synonyms(word):
    try:
        vec = model[word]
        prediction = model.most_similar([vec], topn=5)
        similar_words = [wrd for wrd, _ in prediction]

        return similar_words
    except:
        return None
    
def get_antonyms(word):
    antonyms = [] 

    #syn sets - set of synonyms:
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name()) # taking just names
    return antonyms

if input_word:
    synonyms = get_synonyms(input_word)
    antonyms = get_antonyms(input_word)

    if synonyms:
        st.write(f"Synonyms to {input_word}:")
        for wrd in synonyms:
            st.write(f"{wrd}")
    else:
        st.write("Synonyms not found.")

    st.write("\n---------------------------------\n")

    if antonyms:
        st.write(f"Antonyms to {input_word}:")
        for wrd in antonyms:
            st.write(f"{wrd}")
    else:
        st.write("Antonyms not found.")
