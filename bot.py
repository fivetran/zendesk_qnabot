from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

template = """
You are an assistant tasked with answering questions based on Zendesk chat threads.
Refer to the following chat threads to provide your response.
Keep your answer concise, using a maximum of three sentences.

Question: {question}
Chat Threads: {chat_threads}
Answer:
"""
prompt = PromptTemplate(template)
llm = OpenAI(model="gpt-4")

from llama_index.core import VectorStoreIndex
from pinecone_embeddings import PineconeEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone()
index = pc.Index('zendesk-qna')
vector_store = PineconeVectorStore(pinecone_index=index)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=PineconeEmbedding())
retriever = index.as_retriever()


def get_answer(question):
    chat_threads = [x.get_content() for x in retriever.retrieve(question)]
    response = llm.complete(prompt.format(question=question, chat_threads=chat_threads))
    return response.text


import streamlit as st

st.title("Zendesk - Q&A Bot")

with st.form("my_form"):
    sample_question = "What is the biggest issue with datalakes?"
    question = st.text_area("Enter text:", sample_question)
    submitted = st.form_submit_button("Submit")
    answer = get_answer(question)
    st.info(answer)
