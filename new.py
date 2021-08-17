import streamlit as st

from src.keyword_ex import *

if __name__ == '__main__':

    st.title("Natural Language Processing")

    menu = ["Keyword Extraction", "Topic Modelling", "Sentiment Analysis", "Text Summarization"]
    choice = st.sidebar.selectbox("Natural Language Processing", menu)

    if choice == "Keyword Extraction":

        menu_main = ["keyBERT", "PKE", "RAKE", "YAKE"]
        choice_main = st.selectbox("Unsupervised Keyword Extraction", menu_main)

        if choice_main == "keyBERT":
            type_task = st.radio("Select an Embedding Model", ("Sentence Transformer", "Flair"))
            if type_task == 'Sentence Transformer':
                Sentence_Transformer()
            if type_task == 'Flair': 
                Flair()