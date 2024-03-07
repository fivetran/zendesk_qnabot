from langchain_community.document_loaders.athena import AthenaLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

with open('query.sql', 'r') as q:
    query = q.read()

docs = AthenaLoader(
    query=query,
    database="zendesk",
    s3_output_uri="s3://asimov-datalake-s3/query_results/",
    profile_name="datasharing",
    metadata_columns=["ticket_id", "ticket_subject", "ticket_created_at"]
).load()

print(docs[0].page_content)

embeddings = OpenAIEmbeddings()
vecdb = Chroma.from_documents(docs, embeddings, persist_directory="./db")
vecdb.persist()
