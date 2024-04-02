from langchain_community.document_loaders.athena import AthenaLoader
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec
import time
import os

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_name = 'zendesk-qna'

print("CREATE OR REPLACE INDEX")
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
    index_name,
    dimension=1536,  # dimensionality of text-embedding-ada-002
    metric='dotproduct',
    spec=ServerlessSpec(cloud='aws',region='us-west-2')
)
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

with open('query.sql', 'r') as q:
    query = q.read()

print("DOWNLOADING DOCS FROM DATALAKE")
docs = AthenaLoader(
    query=query,
    database="zendesk",
    s3_output_uri="s3://asimov-datalake-s3/query_results/",
    profile_name="datasharing",
    metadata_columns=["ticket_id", "ticket_subject", "ticket_created_at"]
).load()

print("UPLOADING DOCS TO PINECONE")
embeddings = OpenAIEmbeddings()
index = pc.Index(index_name)
for i in range(0, len(docs), 100):
    print(f"Uploading Batch ({i} to {i+100})")
    batch = docs[i:i+100]
    ids = [str(d.metadata["ticket_id"]) for d in batch]
    payloads = [{**d.metadata, "ticket_details": d.page_content} for d in batch]
    vectors = embeddings.embed_documents([d.page_content for d in batch])

    index.upsert(vectors=zip(ids, vectors, payloads))