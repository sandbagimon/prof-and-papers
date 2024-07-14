import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent

import os
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import QdrantClient


import autogen.runtime_logging

# logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": "runtime.log"})
# print("Logging session ID: " + str(logging_session_id))



config_list = [
    {
        # "model": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",  # the name of your running model
        # "model": "gemma-7b-it",  # the name of your running model
        "model": "Mixtral-8x7b-32768",  # the name of your running model
        # "base_url": "https://5ba7-35-77-94-0.ngrok-free.app/v1",  # the local address of the api
        "base_url": "https://api.groq.com/openai/v1",
        "api_type": "open_ai",
        "api_key": os.getenv("API_KEY"),  # just a placeholder
        # "api_key": os.getenv("API_KEY"),  # just a placeholder
    },

]

qdrant_client = QdrantClient(
    url="https://1d2b3877-7d70-41c6-87d1-21eeea1b2db7.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="ot6n_1Fr64q-xYwDq8eueZ-oSztM1edtsM1X677qXGyGQW4hkDDWsw",
)

# autogen.ChatCompletion.start_logging()



assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)



rag_proxy_agent = QdrantRetrieveUserProxyAgent(
    name="qdrantagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
    retrieve_config={
        "task": "qa",
        # "docs_path": "./real_test_paper.pdf",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "client": qdrant_client,
        # "embedding_model": "BAAI/bge-small-en-v1.5",
        "collection_name": "autogen-test",
        "get_or_create": True,
    },
)

# Always reset the assistant before starting a new conversation.

# We use the ragproxyagent to generate a prompt to be sent to the assistant as the initial message.
# The assistant receives the message and generates a response. The response will be sent back to the ragproxyagent for processing.
# The conversation continues until the termination condition is met, in RetrieveChat, the termination condition when no human-in-loop is no code block detected.

# The query used below is for demonstration. It should usually be related to the docs made available to the agent
assistant.reset()
rag_proxy_agent.retrieve_docs("What is Autogen?", n_results=10, search_string="autogen")
# rag_proxy_agent.initiate_chat(assistant, message=rag_proxy_agent.message_generator, problem="What's autogen?")