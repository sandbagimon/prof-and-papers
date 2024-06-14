import streamlit as st
import pandas as pd
import os
from autogen_test import group_chat_init


st.title('Professors and papers V2 (with more Context Window)')



uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    # temp_file_path = os.path.join("tempDir", uploaded_file.name)
    temp_file_path = os.path.join("tempDir", 'temp.pdf')
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        file_name = uploaded_file.name

    with st.container():
        group_chat_init(temp_file_path)

