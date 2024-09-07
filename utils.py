from pymilvus import connections, utility, Collection

def get_tables(host, token):
    connections.connect(
        alias="default",
        uri=host,
        token=token,
        secure=True
    )

    collection_names = utility.list_collections()

    collections = []
    for name in collection_names:
        collection = Collection(name)

        description = collection.description
        if "fivetran_managed=true" not in description:
            collection.release()
            continue

        collections.append(name)

        collection.release()

    connections.disconnect("default")
    return collections