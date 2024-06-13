from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from Iceberg_reader import IcebergReader

docs = IcebergReader().load_data(
    profile_name="datasharing",
    region="us-west-2",
    namespace="google_drive",
    table="abhijeeth_s_gen_ai_world_of_marketing_conferences",
    metadata_columns_in=['_line', '_fivetran_synced', 'status', 'type_of_attendence', 'priority','dates', 'name', 'location']
)

db = chromadb.PersistentClient(path="./db")
chroma_collection = db.get_or_create_collection("default")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vecdb = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, embed_model=OpenAIEmbedding()
)
