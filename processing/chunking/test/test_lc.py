import sys

from pathlib import Path

# add backend to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))


from processing.chunking.late_chunking import LateChunking
from processing.embeddings import chunk_by_characters, QwenEmbeddingFunction
import unittest
from transformers import AutoTokenizer, AutoModel


class TestLC(unittest.TestCase):
    def test_lc(self):
        text = open("data/extracted/test/resume.md", "r").read()
        late_chunk = LateChunking()
        late_chunk.chunk_text(text)
        query = "what is richard's experience with AI?"

        semantic_results = late_chunk.semantic_retrieval(query, top_k=3)
        for i, chunk_id in enumerate(semantic_results):
            print(f"Rank {i + 1}: {late_chunk.chunks[chunk_id]}")

    def test_qwen(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

        text = open("data/extracted/test/resume.md", "r").read()
        late_chunk = LateChunking(
            tokenizer=tokenizer,
            model=model,
            chunk_method=chunk_by_characters,
            embedding_function=QwenEmbeddingFunction(tokenizer, model),
        )
        late_chunk.chunk_text(text)
        query = "what is richard's experience with AI?"

        semantic_results = late_chunk.semantic_retrieval(query, top_k=3)
        for i, chunk_id in enumerate(semantic_results):
            print(f"Rank {i + 1}: {late_chunk.chunks[chunk_id]}")

    def test_lc_with_chunk_by_sentences(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        text = open("data/extracted/test/resume.md", "r").read()

        def chunk_by_sentences(text, tokenizer):
            return chunk_by_characters(text, tokenizer, chunk_char=["."])

        late_chunk = LateChunking(
            tokenizer=tokenizer,
            model=model,
            chunk_method=chunk_by_sentences,
            embedding_function=QwenEmbeddingFunction(tokenizer, model),
        )
        late_chunk.chunk_text(text)
        query = "what is richard's experience with AI?"

        semantic_results = late_chunk.semantic_retrieval(query, top_k=3)
        for i, chunk_id in enumerate(semantic_results):
            print(f"Rank {i + 1}: {late_chunk.chunks[chunk_id]}")

        bm25_results, bm25_scores = late_chunk.bm25_retrieval(query, top_k=3)
        print("bm25_results: ", bm25_results[0, 0])
        print("bm25_scores: ", bm25_scores[0, 0])
        for i, (chunk_id, score) in enumerate(zip(bm25_results[0], bm25_scores[0])):
            print(f"Rank {i + 1}: {late_chunk.chunks[chunk_id]} (score: {score:.2f})")

        results, semantic_count, bm25_count = late_chunk.retrieve(query, top_k=5)
        for i, result in enumerate(results):
            print(f"Rank {i + 1}: {result['chunk']} (score: {result['score']:.2f})")
            print(f"chunk retrieved from semantic: {result['from_semantic']}")
            print(f"chunk retrieved from bm25: {result['from_bm25']}")
        print("semantic_count: ", semantic_count)
        print("bm25_count: ", bm25_count)


if __name__ == "__main__":
    unittest.main()
