import sys
import math
import streamlit as st
import pandas as pd

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from word_embeddings.tf_idf import tf


# def display_clickable_words(sentence):
#     cleaned_sentence = " ".join([word.lower() for word in sentence.split()])
#     words = set(cleaned_sentence.split())
#     for word in words:
#         if st.button(word):
#             tf_result = tf(cleaned_sentence, word)
#             st.write(f"Term Frequency of {word} is : {tf_result:.4f}")
            
def display_tf_words(sentence):
    cleaned_sentence = " ".join([word.lower() for word in sentence.split()])
    words = set(cleaned_sentence.split())
    results = []
    for word in words:
        tf_result = tf(cleaned_sentence, word)
        results.append({"word": word, "TF": tf_result})
    st.dataframe(pd.DataFrame(results))

def display_idf_words(corpus):
    results = []
    vocab = []
    
    for d in corpus.split(","):
        for t in d.split():
            if t.lower() not in vocab:
                vocab.append(t.lower())
    st.write(f"Vocab: {vocab}")
    for word in vocab:
        docs_w_term = 0
        for document in corpus.split(","):
            cleaned_sentence = " ".join([word.lower() for word in document.split()])
            if word in cleaned_sentence:
                docs_w_term += 1

        if docs_w_term == 0:
            results.append({"word": word, "IDF": 0})
        else:
            results.append({"word": word, "IDF": math.log((len(corpus.split(",")) +1) / (docs_w_term + 1)) + 1})

    st.dataframe(pd.DataFrame(results))

def display_tf_idf_words(corpus):
    results = []
    vocab = []
    idfs = {}
    tfs = {}
    for d in corpus.split(","):
        for t in d.split():
            if t.lower() not in vocab:
                vocab.append(t.lower())
    st.write(f"Vocab: {vocab}")

    for word in vocab:
        count = 0
        for document in corpus.split(","):
            cleaned_sentence = " ".join([word.lower() for word in document.split()])
            tfs[word] = tf(cleaned_sentence, word)
            if word in cleaned_sentence:
                count += 1

        if count == 0:
            idfs[word] = 0
        else:
            idfs[word] = math.log((len(corpus.split(",")) +1) / (count + 1)) + 1
        results.append({"word" : word, "tf-idf" : idfs[word] * tfs[word]})

    st.dataframe(pd.DataFrame(results))

def main():
    st.title("Word Embeddings Demo")

    # Create columns for layout
    options = ["TF", "IDF", "TF-IDF"]
    selection = st.pills("Directions", options, selection_mode="single")
    # Handle "TF" button click
    sentence = st.text_input("Sentence")  
    if selection == 'TF':
        display_tf_words(sentence)

    # Handle "IDF" button click (Placeholder for future implementation)
    if selection == 'IDF':
        display_idf_words(sentence)

    # Handle "TF-IDF" button click (Placeholder for future implementation)
    if selection == 'TF-IDF':
        display_tf_idf_words(sentence)

if __name__ == "__main__":
    main()









