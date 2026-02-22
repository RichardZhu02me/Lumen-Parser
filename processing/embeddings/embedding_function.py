from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
import json
from .qwen import get_query_embedding


@register_embedding_function
class TransformersEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model=None):
        self.model = model
        if model is None:
            self.model = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v3", trust_remote_code=True
            )

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = self.model.encode(input)
        return embeddings

    @staticmethod
    def name() -> str:
        return "my-transformers-embedding-function"

    def get_config(self) -> Dict[str, Any]:
        return self.model.config.to_dict()

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return TransformersEmbeddingFunction(config["model"])


@register_embedding_function
class QwenEmbeddingFunction(EmbeddingFunction):
    def __init__(self, tokenizer=None, model=None):
        self.tokenizer = tokenizer
        self.model = model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
            )
        if model is None:
            self.model = AutoModel.from_pretrained(
                "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
            )

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = get_query_embedding(self.tokenizer, self.model, input)
        return embeddings

    @staticmethod
    def name() -> str:
        return "my-transformers-embedding-function"

    def get_config(self) -> Dict[str, Any]:
        return self.model.config.to_dict()

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return TransformersEmbeddingFunction(config["model"])


if __name__ == "__main__":
    my_ef = TransformersEmbeddingFunction()
    embeddings = my_ef(["hello", "world"])
    print(type(embeddings[1]))
    print(json.dumps(my_ef.get_config()))
