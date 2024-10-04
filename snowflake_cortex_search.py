from typing import Any, List, Iterable, Optional

from langchain_core.vectorstores.base import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from snowflake.core import Root
from snowflake.snowpark import Session



class SearchSnowflakeCortex(VectorStore):

    def __init__(
            self,
            session_builder_conf,
            snowflake_database,
            snowflake_schema,
            snowflake_cortex_search_service
    ):
        self.session_builder_conf = session_builder_conf
        self.snowflake_database: str = snowflake_database
        self.snowflake_schema: str = snowflake_schema
        self.snowflake_cortex_search_service: str = snowflake_cortex_search_service

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        session = Session.builder.configs(self.session_builder_conf).create()

        root = Root(session)
        search_service = root.databases[self.snowflake_database].schemas[self.snowflake_schema].cortex_search_services[
            self.snowflake_cortex_search_service]

        desc_result = session.sql(f"DESC CORTEX SEARCH SERVICE {self.snowflake_cortex_search_service}").collect()[0]

        search_column = desc_result.search_column
        columns = desc_result.columns.split(",")

        search_resp = search_service.search(
            query=query,
            columns=columns,
            limit=k
        )

        relevant_docs = []
        for row in search_resp.results:
            metadata = {
                col: value
                for col, value in row.items()
                if col != search_column
            }
            doc = Document(page_content=row[search_column], metadata=metadata)
            relevant_docs.append(doc)

        session.close()

        return relevant_docs

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ):
        raise NotImplementedError(f"`from_texts` has not been implemented")