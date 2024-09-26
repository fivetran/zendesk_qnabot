from pymilvus import connections, utility, Collection
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings


def get_collections(host, token):
    connections.connect(
        alias="default",
        uri=host,
        token=token,
        secure=True
    )

    collection_names = utility.list_collections()

    filtered_collections = {}
    for name in collection_names:

        collection = Collection(name)
        description = collection.description
        if "fivetran_managed=true" in description:
            filtered_collections[name] = {"primary_field" : collection.primary_field.name}
        collection.release()

    connections.disconnect("default")
    return filtered_collections

def get_vector_stores(host, token, selected_collections, collections, openai_api_key, embedding_model):
    connection_args = {
        "uri": host,
        "token": token,
        "secure": True
    }

    embedding_function = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=openai_api_key
    )

    vector_stores = {}
    for collection_name, collection_metadata in collections.items():
        if collection_name not in selected_collections:
            print("Skipping collection : ", collection_name)
            continue
        vector_stores[collection_name] = Milvus(
            embedding_function=embedding_function,
            connection_args=connection_args,
            collection_name=collection_name,
            text_field="original_text",
            vector_field="vector",
            primary_field=collection_metadata['primary_field']
        )

    return vector_stores