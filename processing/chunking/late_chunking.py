import sys
import os

# add backend to python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from transformers import AutoModel, AutoTokenizer
from processing.embeddings.chunked_pooling import chunked_pooling, chunk_by_sentences
import numpy as np
import bm25s
from processing.embeddings.embedding_function import TransformersEmbeddingFunction


class LateChunking:
    def __init__(
        self,
        tokenizer=None,
        model=None,
        chunk_method: callable = chunk_by_sentences,
        embedding_function: callable = None,
    ):
        # load model and tokenizer
        self.tokenizer = tokenizer
        self.model = model
        if not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "jinaai/jina-embeddings-v3", trust_remote_code=True
            )
        if not model:
            self.model = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v3", trust_remote_code=True
            )
        self.embedding_function = embedding_function
        if not embedding_function:
            self.embedding_function = TransformersEmbeddingFunction(model=self.model)
        self.chunk_method = chunk_method

        self.chunks = []
        self.chunk_embeddings = []

    def chunk_text(self, text: str):
        # determine chunks
        chunks, span_annotations = self.chunk_method(text, self.tokenizer)
        self.chunks.extend(chunks)

        # chunk afterwards (context-sensitive chunked pooling)
        inputs = self.tokenizer(text, return_tensors="pt")
        model_output = self.model(**inputs)
        self.chunk_embeddings.extend(
            chunked_pooling(model_output, [span_annotations])[0]
        )

        # Create the BM25 model and index the corpus (must be recreated from scratch)
        self.retriever = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(self.chunks)
        self.retriever.index(corpus_tokens)

    ## retrieve function using late chunking
    # returns the top_k chunk ids via late chunking semantic similarity
    def semantic_retrieval(
        self, query, top_k=5
    ) -> list[int]:  # Default value for document to return is 5
        # Convert the input query into a numerical embedding vector using the pre-trained embedding model.
        # query_embedding = self.model.encode(query)
        query_embedding = self.embedding_function(query)

        def cos_sim(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        # Compute the cosine similarity between the query embedding (reshaped into a 2D array) and all stored embeddings.
        # This returns an array of similarity scores for each stored document.
        similarity_score = [
            cos_sim(query_embedding, chunk_embed)
            for chunk_embed in self.chunk_embeddings
        ]

        # Sort the indices of documents in descending order of similarity (highest similarity first).
        similarity_indices = np.argsort(-1 * np.array(similarity_score).flatten())
        # Select the first 'top_k' indices from the sorted list, which correspond to the most similar documents.
        top_k_indices = similarity_indices[:top_k]

        return top_k_indices

    # returns the top_k chunk ids via bm25 keyword search
    def bm25_retrieval(self, query, top_k=5) -> list[int]:
        # Create the BM25 model and index the corpus
        results, scores = self.retriever.retrieve(bm25s.tokenize(query), k=top_k)
        return results, scores

    def retrieve(self, query, top_k=5, semantic_weight=0.8, bm25_weight=0.2):
        # Get semantic results
        semantic_results = [int(i) for i in self.semantic_retrieval(query, top_k=top_k)]
        # Create a dictionary of semantic ranks
        semantic_rank = {}
        print(semantic_results)
        for i, chunk_id in enumerate(semantic_results):
            semantic_rank[chunk_id] = i
        print("semantic_ranks: ", semantic_rank)
        # Get BM25 results
        bm25_results, bm25_scores = self.bm25_retrieval(query, top_k=top_k)
        bm25_results = [int(i) for i in bm25_results[0]]
        # Create a dictionary of BM25 ranks
        bm25_rank = {}
        for i, chunk_id in enumerate(bm25_results):
            bm25_rank[chunk_id] = i
        print("bm25_ranks: ", bm25_rank)
        # Combine results
        combined_results = list(set(semantic_results + bm25_results))
        # Remove duplicates
        scores = {}
        for chunk_id in combined_results:
            score = 0
            if chunk_id in semantic_results:
                score += semantic_weight * semantic_rank[chunk_id]
            if chunk_id in bm25_results:
                score += bm25_weight * bm25_rank[chunk_id]
            scores[chunk_id] = score

        # Return top_k results
        sorted_chunk_ids = sorted(
            combined_results, key=lambda idx: (scores[idx], idx), reverse=True
        )
        print("sorted_chunk_ids: ", sorted_chunk_ids)

        final_results = []
        semantic_count = 0
        bm25_count = 0
        print("combined_results: ", combined_results)
        print("chunk size: ", len(self.chunks))
        for chunk_id in sorted_chunk_ids[:top_k]:
            is_from_semantic = chunk_id in semantic_results
            is_from_bm25 = chunk_id in bm25_results
            final_results.append(
                {
                    "chunk": self.chunks[chunk_id],
                    "score": scores[chunk_id],
                    "from_semantic": is_from_semantic,
                    "from_bm25": is_from_bm25,
                }
            )

            if is_from_semantic and not is_from_bm25:
                semantic_count += 1
            elif is_from_bm25 and not is_from_semantic:
                bm25_count += 1
            else:  # it's in both
                semantic_count += 0.5
                bm25_count += 0.5
        return final_results, semantic_count, bm25_count


if __name__ == "__main__":
    text = open("data/extracted/test/resume.md", "r").read()
    late_chunk = LateChunking()
    late_chunk.chunk_text(text)
    query = "what is richard's experience with AI?"

    semantic_results = late_chunk.semantic_retrieval(query, top_k=3)
    print("semantic_results: ", semantic_results)
    for i, chunk_id in enumerate(semantic_results):
        print(f"Rank {i + 1}: {late_chunk.chunks[chunk_id]}")

    bm25_results, bm25_scores = late_chunk.bm25_retrieval(query, top_k=3)
    print("bm25_results: ", bm25_results[0, 0])
    print("bm25_scores: ", bm25_scores[0, 0])
    for i, (chunk_id, score) in enumerate(zip(bm25_results[0], bm25_scores[0])):
        print(f"Rank {i + 1}: {chunk_id} (score: {score:.2f})")

    results, semantic_count, bm25_count = late_chunk.retrieve(query, top_k=5)
    print(results)
    print("semantic_count: ", semantic_count)
    print("bm25_count: ", bm25_count)
