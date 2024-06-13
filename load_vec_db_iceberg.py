from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

from pinecone_embeddings import PineconeEmbedding
from llama_index.readers.iceberg import IcebergReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time

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

docs = IcebergReader().load_data(
    profile_name="datasharing",
    region="us-west-2",
    namespace="wine_country_dataset",
    table="wine_stats",
    metadata_columns=[
        "_file",
        "_line",
        "_modified",
        "_fivetran_synced",
    ]
)

pipeline.run(documents=docs)
