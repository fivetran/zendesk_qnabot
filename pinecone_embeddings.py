from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from typing import Any, List
from pinecone import Pinecone

class PineconeEmbedding(BaseEmbedding):
    _pc: Pinecone = PrivateAttr()
    def __init__(
            self,
            **kwargs: Any,
    ) -> None:
        self._pc = Pinecone()
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "pinecone"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={
                "input_type": "query",
                "truncate": "END"
            }
        )
        return embeddings.data[0].values

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[text],
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )
        return embeddings.data[0].values

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = [x.values for x in self._pc.inference.embed(
            model="multilingual-e5-large",
            inputs=texts,
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        ).data]
        return embeddings
