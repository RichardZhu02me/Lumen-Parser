import sys
import os

# add backend to python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import unittest
from processing.vector_store import VectorStore
from chromadb.api.types import Document


class TestVectorStore(unittest.TestCase):
    def test_ingest_documents(self):
        vs = VectorStore(path="data/chromadb/test")
        vs.delete_collection("test")
        vs.ingest_documents(
            ["hello", "I am a sigma", "siggy siggy yay"], collection_name="test"
        )

        self.assertEqual(vs.list_collections()[0].name, "test")
        self.assertEqual(vs.get_collection("test").count(), 3)

        collection = vs.get_collection("test")
        records = collection.get()

        self.assertTrue(
            all(
                record in records["documents"]
                for record in ["hello", "I am a sigma", "siggy siggy yay"]
            )
        )
        self.assertTrue(records["embeddings"] is None)

    def test_query_collection(self):
        vs = VectorStore(path="data/chromadb/test")
        vs.delete_collection("test")
        dummy_content = ["hello", "I am a sigma", "siggy siggy yay"]
        dummy_metadata = [{"metadata number": i} for i in range(3)]
        vs.ingest_documents(
            dummy_content, collection_name="test", metadatas=dummy_metadata
        )

        response = vs.query_collection("test", "hi how are you")

        # records = zip(
        #     response["ids"],
        #     response["documents"],
        #     response["metadatas"],
        #     response["distances"],
        # )
        # for id, document, metadata, distance in records:
        #     print(id, document, metadata, distance)

        ## first document is most similar
        self.assertTrue(
            response["distances"][0] <= distance for distance in response["distances"]
        )

        ## embeddings are not stored
        collection = vs.get_collection("test")
        records = collection.get()
        self.assertTrue(records["embeddings"] is None)

    def test_example_case(self):
        import json

        with open("data/processed/test/resume_child_chunks.json") as json_data:
            chunks = json.loads(json_data.read())

        content = [chunk["content"] for chunk in chunks]
        metadata = [chunk["metadata"] for chunk in chunks]
        for chunk in chunks:
            self.assertEqual(type(chunk["metadata"]["page"]), type(None))
            self.assertEqual(type(chunk["metadata"]["level"]), type(None))

        self.assertEqual(len(content), len(metadata))
        vs = VectorStore(path="data/chromadb/test")
        vs.delete_collection("test")
        vs.ingest_documents(content, collection_name="test", metadatas=metadata)

        # print(vs.get_collection("test").get())

        response = vs.query_collection("test", "what is richard's experience with AI?")

        records = zip(
            response["ids"],
            response["documents"],
            response["metadatas"],
            response["distances"],
        )
        # for id, document, metadata, distance in records:
        for record in records:
            print(record)


if __name__ == "__main__":
    unittest.main()
