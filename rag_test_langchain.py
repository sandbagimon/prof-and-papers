import os

from langchain_qdrant import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import streamlit as st




def langchain_rag(pdf):
    load_dotenv()
    st.text('Loading Embedding...')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    st.text("Embedding loaded.")
    url = os.getenv("VD_URL")
    api_key = os.getenv("VD_API_KEY")
    st.text("Loading Documents")
    loader = PDFMinerLoader(pdf)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Qdrant.from_documents(documents=splits, embedding=embeddings, url=url, api_key=api_key)
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = hub.pull("imon/research_rag_prompt")
    st.text("Document Loaded..")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        api_key=os.getenv("API_KEY"),  # Optional if not set as an environment variable
    )
    rag_chain = (
            # {"context": retriever | format_docs, "question": RunnablePassthrough()}
            {"context": retriever | format_docs}
            | prompt
            | llm
            | StrOutputParser()
    )

    # rag_chain.invoke("Summarize the paper")

    st.write_stream(rag_chain.stream("Show me the writing analysis of the paper"))



# doc_store = Qdrant.from_texts(
#     texts=text,
#     embedding=embeddings,
#     url=url,
#     api_key=api_key,
#     collection_name="texts"
# )


