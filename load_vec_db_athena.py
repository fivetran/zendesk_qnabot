from llama_index.readers.athena import AthenaReader

from llama_index.core import SQLDatabase
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
import pandas as pd
import json
import chromadb
import os

with open('query.sql', 'r') as q:
    query = q.read()


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

db = chromadb.PersistentClient(path="./db")
chroma_collection = db.get_or_create_collection("default")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vecdb = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, embed_model=OpenAIEmbedding()
)
