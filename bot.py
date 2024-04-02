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
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVS
from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

def get_answer(question):
    index = pc.Index('zendesk-qna')
    db = PineconeVS(index, OpenAIEmbeddings(), "ticket_details")
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
