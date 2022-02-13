import matplotlib.pyplot as plt
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(
    page_icon='📖',
    initial_sidebar_state='expanded'
)

st.title('Austen or Poe?')

st.write('Use the sidebar to select a page to view.')

page = st.sidebar.selectbox(
    'Page',
    ('About', 'EDA', 'Make a prediction')
)

if page == 'About':
    st.subheader('About this project')
    st.write('''
This is a Streamlit app that hosts my Poe vs. Austen model.

The best model I found was....

You can get in touch with me on these websites....

etc.
    ''')
elif page == 'EDA':
    # header
    st.subheader('Exploratory Data Analysis')
    st.write('''The model is trained on paragraphs.
    
Below you can find....''')
    

elif page == 'Make a prediction':
    st.subheader('Which author do you write like?')

    st.write('''
Enter some text to make a prediction!

The text might visually cut off, but you can write up to 1,000 characters.
    ''')

    # filepath different b/c solution code
    with open('../models/author_pipe.pkl', 'rb') as pickle_in:
        pipe = pickle.load(pickle_in)

    your_text = st.text_input(
        label='Please enter some text:',
        value='It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife',
        max_chars=1000)
    predicted_author = pipe.predict([your_text])[0]

    st.subheader('Results:')
    st.write(f'You write more like {predicted_author.title()}.')