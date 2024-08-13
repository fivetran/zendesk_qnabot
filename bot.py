import snowflake.connector
from langchain_core.documents import Document
import os

from openai import OpenAI

client = OpenAI()

def get_embedding(content):
    try:
        return client.embeddings.create(
            input=[content],
            model="text-embedding-ada-002"
        ).data[0].embedding
    except Exception as e:
      if "maximum context length" in str(e):
        print(f"Error in get_embedding: Input exceeds maximum context length")
        return [-1.1] * 3072
      else:
        raise
def get_relevant_docs(question: str):
        conn = snowflake.connector.connect(
            account= "FIVETRAN",
            user= os.getenv("SNOWFLAKE_USER"),
            password= os.getenv("SNOWFLAKE_PASSWORD"),
            role= "ACCOUNTADMIN",
            database= "FIVETRAN_DATABASE",
            warehouse= "FIVETRAN_WAREHOUSE",
            schema= "PUBLIC_FIVETRAN_CHAT",
        )

        relevant_docs = []
        question_embedding = get_embedding(question)
        for (document_id, chunk_index, chunk, similarity) in conn.cursor().execute(f"SELECT document_id, chunk_index, chunk, VECTOR_COSINE_SIMILARITY(embedding, {question_embedding}::VECTOR(FLOAT, 1536)) AS similarity FROM demo_pdfs_embedding ORDER BY similarity DESC LIMIT 10"):
            metadata = {
                "document_id": document_id,
                "chunk_index": chunk_index,
            }
            doc = Document(page_content=chunk, metadata=metadata)
            relevant_docs.append(doc)

        return relevant_docs


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """
You are an assistant tasked with answering questions based on credit card statements.
Refer to the following parts of the credit card statement to provide your response.
Keep your answer concise, using a maximum of three sentences.

Question: {question}
Credit Card Statements: {credit_card_statements}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4-0125-preview")

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings


def get_answer(question):
    relevant_docs = get_relevant_docs(question)
    credit_card_statements = "\n".join([doc.page_content for doc in relevant_docs])

    inputs = {"credit_card_statements": credit_card_statements, "question": question}
    rag_chain = (RunnablePassthrough() | prompt | llm | StrOutputParser())
    return rag_chain.invoke(inputs)


import streamlit as st

st.title("Any PDF - Q&A Bot")

with st.form("my_form"):
    sample_question = "How much money did i spend charging my Tesla?"
    question = st.text_area("Enter text:", sample_question)
    submitted = st.form_submit_button("Submit")
    answer = get_answer(question)
    st.info(answer)
