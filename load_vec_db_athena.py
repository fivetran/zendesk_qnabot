from llama_index.core import SQLDatabase
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from pinecone_embeddings import PineconeEmbedding
from llama_index.readers.athena import AthenaReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import time
import json
import os

with open('query.sql', 'r') as q:
    query = q.read()

pc = Pinecone()
index_name = 'zendesk-qna'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=1024,  # dimensionality of multilingual-e5-large
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=index)
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=200),
        PineconeEmbedding(),
    ],
    vector_store=vector_store,
)


engine = AthenaReader().create_athena_engine(
    aws_access_key=os.getenv("aws_access_key"),
    aws_secret_key=os.getenv("aws_secret_key"),
    aws_region=os.getenv("aws_region"),
    s3_staging_dir="s3://asimov-datalake-s3/query_results/",
    database="zendesk",
)

res =  SQLDatabase(engine=engine).run_sql(query)[1]
docs = pd.DataFrame(res["result"], columns=res["col_keys"])
docs = json.loads(docs.to_json(orient='records'))
docs = [Document(text=e["ticket_details"], metadata=e) for e in docs]

pipeline.run(documents=docs)