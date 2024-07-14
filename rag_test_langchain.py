import os

from langchain_qdrant import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from langchain.retrievers.multi_query import MultiQueryRetriever

from dotenv import load_dotenv
import streamlit as st


def langchain_rag(pdf):
    load_dotenv()
    st.text('Loading Embedding...')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
        # model_name="thenlper/gte-large"
    )
    st.text("Embedding loaded.")
    url = os.getenv("VD_URL")
    api_key = os.getenv("VD_API_KEY")
    st.text("Loading Documents")
    ###########################for faster local test

    # vectorstore= Qdrant.from_existing_collection(embedding=embeddings, collection_name="test_text", url=url, api_key=api_key)
    ################################
    loader = PDFMinerLoader(pdf)
    # loader = LLMSherpaFileLoader(pdf)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Qdrant.from_documents(documents=splits, embedding=embeddings, url=url, api_key=api_key,collection_name="test_text",force_recreate=True)
    # vectorstore = Qdrant.from_documents(documents=docs, embedding=embeddings, url=url, api_key=api_key,collection_name="test_text",force_recreate=True)

    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        # model="gemma2-9b-it",
        api_key=os.getenv("API_KEY"),  # Optional if not set as an environment variable
    )

    # Retrieve and generate using the relevant snippets of the blog.

    ####################retriever test
    retriever= MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )

    # retriever = vectorstore.as_retriever()
    #############################################
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = hub.pull("imon/research_rag_prompt")
    st.text("Document Loaded..")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            # {"context": retriever | format_docs}
            | prompt
            | llm
            | StrOutputParser()
    )

    # rag_chain.invoke("Summarize the paper")
    system_prompt = """
    ## Criteria 

* **Clarity and precision**

- Is the language used clear and precise? 

- Are the key concepts and arguments easy to understand?

 - Are technical terms and jargon appropriately defined and explained?

* **Coherence and flow**

- Is the paper logically organized?

 - Do the ideas and arguments flow smoothly from one section to the next?

 - Are there effective transitions between paragraphs and sections?

* **Organization and structure** 

- Does the paper have a clear and effective structure (introduction, literature review, methodology, results, discussion, conclusion)? 

- Are the sections well-defined and appropriately titled? 

- Is there a logical progression of ideas and arguments?

* **Grammar and syntax** 

- Is the paper free of grammatical errors and typographical mistakes? 

- Are sentence structures varied and appropriate for academic writing? 

- Is the use of punctuation correct and consistent?

* **Adherence to academic standards**

- Does the paper follow the appropriate citation style (e.g., APA, MLA, Chicago)? 

- Are all sources properly cited and referenced?

- Does the paper demonstrate critical thinking and originality?

* **Overall impact** 

- Does the paper effectively communicate its research findings? 

- Is the writing engaging and persuasive?

- Are the conclusions well-supported by the data and arguments presented?

## Provide detailed feedback 

For each criterion, provide specific feedback on what the author does well and what needs improvement. Please point out exact locations in the paper where improvements are needed, using page numbers and paragraph numbers. 

## Ensure results match conclusions 

Additionally, please verify that the results presented in the paper align with the conclusions drawn by the author. Check if the data supports the claims made and if the author has properly interpreted the findings. 

## Format your feedback 

#### Clarity and precision

* [Your feedback here, including chain of thinking] 

#### Coherence and flow

 * [Your feedback here, including chain of thinking] 

#### Organization and structure 

* [Your feedback here, including chain of thinking] 

#### Grammar and syntax 

* [Your feedback here, including chain of thinking] 

#### Adherence to academic standards 

* [Your feedback here, including chain of thinking] 

#### Overall impact 

* [Your feedback here, including chain of thinking] 

#### Results matching conclusions 

* [Verify that the results support the conclusions and provide feedback on any discrepancies] 

## Additional suggestions 

If you have any additional suggestions for improvement that don't fit into the above categories, please include them in your feedback. 

## Chain of thinking 

THIS SECTION DOES NOT PROVIDE TO THE RESPONSE

When providing feedback, please outline your chain of thinking, showing how you arrived at your conclusions. This means:

 * **Identifying specific text snippets or sections** that support your feedback 

* **Explaining how these snippets or sections relate to the evaluation criteria** 

* **Detailing any logical connections or inferences** you made to reach your conclusions """
    st.write(rag_chain.invoke("""**Clarity and precision**

- Is the language used clear and precise? 

- Are the key concepts and arguments easy to understand?

 - Are technical terms and jargon appropriately defined and explained?


 """))
    st.write(rag_chain.invoke("""
    * **Coherence and flow**

- Is the paper logically organized?

 - Do the ideas and arguments flow smoothly from one section to the next?

 - Are there effective transitions between paragraphs and sections?

  You MUST provide the exact context to the reponse.
  [Context]
    """))
    st.write(rag_chain.invoke("""
    * **Organization and structure**

- Does the paper have a clear and effective structure (introduction, literature review, methodology, results, discussion, conclusion)?

- Are the sections well-defined and appropriately titled?

- Is there a logical progression of ideas and arguments?

 You MUST provide the exact context to the reponse.
 [Context]
    """))


    # vectorstore.adelete(0)

# doc_store = Qdrant.from_texts(
#     texts=text,
#     embedding=embeddings,
#     url=url,
#     api_key=api_key,
#     collection_name="texts"
# )
