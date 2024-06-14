import autogen
import re
from autogen import ConversableAgent
from autogen import AssistantAgent
from langchain_community.document_loaders import PDFMinerLoader
from autogen import Agent
from pprint import pprint
import os
from dotenv import load_dotenv
import nltk
import streamlit as st
import pandas as pd

load_dotenv()


config_list = [
    {
        "model": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",  # the name of your running model
        "base_url": os.getenv("BASE_URL"),  # the local address of the api
        # "api_type": "open_ai",
        "api_key": os.getenv("API_KEY"),  # just a placeholder
    },

]

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


critirias = {
    "Introduction": "You are a Research Paper criteria marker. Your job is to evaluate the paper provided by context based on specific criteria. You only comment on YOUR criterions. For each response, provide a score and feedback for each criterion. The feedback MUST refer to the paper. The criteria are as follows:",
    "Research Quality": grouped_data.loc[grouped_data['Input'] == 'Research Quality', 'Prompt'].values[0],

    "Writing and Presentation": grouped_data.loc[grouped_data['Input'] == 'Writing and Presentation', 'Prompt'].values[0],
    "Impact and Relevance": grouped_data.loc[grouped_data['Input'] == 'Impact and Relevance', 'Prompt'].values[0],
    "Ethics and Validity": grouped_data.loc[grouped_data['Input'] == 'Ethics and Validity', 'Prompt'].values[0],
    "OR_Introduction": "You are an overall recommendation marker. Your job is to evaluate responses based on the scores provided by other experts. For each response, calculate an overall score and provide a brief recommendation. You need to show your score calculation.",
    "Overall Recommendation": grouped_data.loc[grouped_data['Input'] == 'Overall Recommendation', 'Prompt'].values[0],
    "OR_test": "Calculate the total score given by each expert"
}

reinforcement = "\n Remember that you need to provide detailed feedback on the given mark. The feedback should be based on context."


RQ_system_prompt = ""


RQ_bot = AssistantAgent(
    name="Research Quality Assistant",
    llm_config={"config_list": config_list},
    system_message=critirias["Introduction"] + critirias["Research Quality"] + reinforcement,
    human_input_mode="NEVER",  # Never ask for human input.
)


user_proxy_auto = autogen.UserProxyAgent(
    name="User_Proxy_Auto",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="ALWAYS",  # ask human for input at each step
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)



def multi_agent_chat_init(pdf):
    print(pdf)
    loader = PDFMinerLoader(pdf)
    data = loader.load()
    text = data[0].page_content
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ')
        return text.strip()
    def merge_paragraphs(text):
        paragraphs = text.split('\n\n')
        merged_text = ' '.join(paragraphs)
        return merged_text

    text = merge_paragraphs(clean_text(text))

    tasks = "Evaluate the following paper "+text

    chat_results = autogen.initiate_chats(
        [
            {
                "sender": user_proxy_auto,
                "recipient": RQ_bot,
                "message": tasks,
                "clear_history": True,
                "silent": False,
                "summary_method": "reflection_with_llm",
                "max_turns":1,
            },
            # {
            #     "sender": user_proxy_auto,
            #     "recipient": research_assistant,
            #     "message": financial_tasks[1],
            #     "max_turns": 2,  # max number of turns for the conversation (added for demo purposes, generally not necessarily needed)
            #     "summary_method": "reflection_with_llm",
            # },
            # {
            #     "sender": user_proxy,
            #     "recipient": writer,
            #     "message": writing_tasks[0],
            #     "carryover": "I want to include a figure or a table of data in the blogpost.",  # additional carryover to include to the conversation (added for demo purposes, generally not necessarily needed)
            # },
        ]
    )
    with st.chat_message("Research Quality"):
        st.markdown(chat_results[0].chat_history[1]['content'])
if __name__ == "__main__":
    multi_agent_chat_init("test_paper_3.pdf")