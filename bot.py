from langchain_community.document_loaders.athena import AthenaLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def resync_vector_db():
    with open('query.sql', 'r') as q:
        query = q.read()

    docs = AthenaLoader(
        query=query,
        database="zendesk",
        s3_output_uri="s3://asimov-datalake-s3/query_results/",
        profile_name="datasharing",
        metadata_columns=["ticket_id", "ticket_subject", "ticket_created_at"]
    ).load()

    embeddings = OpenAIEmbeddings()
    vecdb = Chroma.from_documents(docs, embeddings, persist_directory="./db")
    vecdb.persist()


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """
You are an assistant tasked with answering questions based on Zendesk chat threads.
Refer to the following chat threads to provide your response.
Keep your answer concise, using a maximum of three sentences.

Question: {question}
Chat Threads: {chat_threads}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4-0125-preview")

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_answer(question):
    db = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
    inputs = {"chat_threads": db.as_retriever(), "question": RunnablePassthrough()}
    rag_chain = (inputs | prompt | llm | StrOutputParser())
    return rag_chain.invoke(question)


import streamlit as st

st.title("Zendesk - Q&A Bot")

with st.form("my_form"):
    sample_question = "What is the biggest issue with datalakes?"
    question = st.text_area("Enter text:", sample_question)
    submitted = st.form_submit_button("Submit")
    answer = get_answer(question)
    st.info(answer)
