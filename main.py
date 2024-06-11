import streamlit as st
import pandas as pd
import os
from autogen_test import group_chat_init

st.title('Professors and papers')



uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    # temp_file_path = os.path.join("tempDir", uploaded_file.name)
    temp_file_path = os.path.join("tempDir", 'temp.pdf')
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        file_name = uploaded_file.name

    with st.container():
        group_chat_init(temp_file_path)
else:
    with st.container():
        st.header('Marking Critiria')
        df = pd.read_excel("IPO.xlsx")
        filtered_data = df[['Criterias', 'Input', 'Prompt', 'A Possible Outcome']].dropna(subset=['Prompt'])
        filtered_data['Criterias'] = filtered_data['Criterias'].fillna(method='ffill')
        grouped_data = filtered_data.groupby('Criterias').agg({
            'Input': lambda x: list(x.dropna()),
            'Prompt': lambda x: list(x.dropna()),
            'A Possible Outcome': lambda x: list(x.dropna())
        }).reset_index()
        grouped_data['Input'] = grouped_data['Input'].apply(lambda x: x[0] if x else '')
        grouped_data['Prompt'] = grouped_data['Prompt'].apply(lambda x: '\n'.join(x))
        st.table(grouped_data)
    # df = group_chat_init("tempDir\\"+uploaded_file.name)



