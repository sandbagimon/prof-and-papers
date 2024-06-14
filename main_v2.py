import streamlit as st
import pandas as pd
import os
from multi_purpose_agent import multi_agent_chat_init


st.title('Professors and papers V2 (with more Context Window Also VERY VERY SLOW)')



uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    # temp_file_path = os.path.join("tempDir", uploaded_file.name)
    temp_file_path = os.path.join("tempDir", 'temp.pdf')
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        file_name = uploaded_file.name

    with st.container():
        multi_agent_chat_init(temp_file_path)

