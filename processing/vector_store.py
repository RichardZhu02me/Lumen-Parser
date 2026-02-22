import sys
import os

# add backend to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromadb import PersistentClient, Documents
from chromadb.api.client import Client
from uuid import uuid4
from pathlib import Path
from processing.embeddings import TransformersEmbeddingFunction

from typing import List, Union
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.base_types import MetadataListValue, SparseVector

CHROMA_PATH = Path("data/chromadb")
DEFAULT_COLLECTION = "database_collection"


class VectorStore:
    def __init__(
        self, path=CHROMA_PATH, embedding_function=TransformersEmbeddingFunction()
    ):
        self.client = PersistentClient(path=path)
        self.embedding_function = embedding_function

    def create_collection(self, collection, **kwargs):
        collection = self.client.create_collection(
            name=collection, embedding_function=self.embedding_function, **kwargs
        )
        return collection

    def get_or_create_collection(self, collection, **kwargs):
        collection = self.client.get_or_create_collection(
            name=collection, embedding_function=self.embedding_function, **kwargs
        )
        return collection

    def list_collections(self, **kwargs) -> List[str]:
        return self.client.list_collections(**kwargs)

    ## ingest documents into the vector store, if no collection is created, returns error
    def ingest_documents(
        self,
        documents: Documents,
        collection_name: str = DEFAULT_COLLECTION,
        ids: List[str] = None,
        hnsw: dict = {"hnsw:space": "cosine"},
        **kwargs,
    ):
        if not ids:
            ids = [str(uuid4()) for _ in range(len(documents))]

        if "metadatas" in kwargs:
            for i, metadata in enumerate(kwargs["metadatas"]):
                kwargs["metadatas"][i] = {
                    k: v for k, v in metadata.items() if v is not None
                }

        collection = self.get_or_create_collection(collection_name, metadata=hnsw)
        return collection.add(documents=documents, ids=ids, **kwargs)

    def ingest_embeddings(
        self,
        embeddings: list[np.ndarray],
        collection_name: str = DEFAULT_COLLECTION,
        ids: List[str] = None,
        hnsw: dict = {"hnsw:space": "cosine"},
        **kwargs,
    ):
        if not ids:
            ids = [str(uuid4()) for _ in range(len(embeddings))]

        collection = self.get_or_create_collection(collection_name, metadata=hnsw)
        return collection.add(embeddings=embeddings, ids=ids, **kwargs)

    def get_collection(self, collection_name, **kwargs) -> Collection:
        return self.client.get_collection(collection_name, **kwargs)

    def delete_collection(self, collection_name, **kwargs):
        try:
            response = self.client.delete_collection(collection_name, **kwargs)
            return response
        except Exception as e:
            print(e)
            return None

    def query_collection(self, collection_name, query: str, **kwargs):
        collection = self.get_collection(collection_name, **kwargs)
        query_embedding = self.embedding_function(query)
        return collection.query(query_embeddings=query_embedding, **kwargs)


if __name__ == "__main__":
    client = VectorStore()
    print(client.list_collections())
    # text = open('data/extracted/output.md', 'r').read()
    docs = ["hello", "I am a sigma", "siggy siggy yay"]
    response = client.ingest_documents(
        documents=docs, collection_name=DEFAULT_COLLECTION
    )
    print(response)

    collection = client.get_collection(DEFAULT_COLLECTION)
    # print(collection)

    query_response = client.query_collection(
        collection_name=DEFAULT_COLLECTION, query="hello"
    )
    print(query_response)
