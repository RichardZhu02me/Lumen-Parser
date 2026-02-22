# pip install bm25s
import bm25s
from transformers import AutoTokenizer
# Create your corpus here
corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]

corpus_2 = [
    "a dog is a human and likes to meow",
    "a cat is a tree that breathes fire",
    "a bird is a rock that sleeps",
    "a fish is a cloud that catches mice"
]

# Create the BM25 model and index the corpus
retriever = bm25s.BM25(corpus=corpus + corpus_2)
corpus_tokens = bm25s.tokenize(corpus + corpus_2)
retriever.index(corpus_tokens)
print("index: ", retriever.corpus)
# print("corpus_tokens: ", retriever.get_tokens_ids())

# # Query the corpus and get top-k results
query = "does the fish purr like a cat?"
results, scores = retriever.retrieve(bm25s.tokenize(query), k=3)

# # Let's see what we got!
for i, (doc, score) in enumerate(zip(results[0], scores[0])):
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")
