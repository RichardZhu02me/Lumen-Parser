import sys
import os

# add backend to python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


import unittest
from processing.schemas import Chunk, Metadata
from pydantic import ValidationError


class TestMetadata(unittest.TestCase):
    def test_metadata(self):
        m1 = Metadata(section="test", header="test", level=1, page=1).model_dump()
        self.assertEqual(m1["section"], "test")
        self.assertEqual(m1["header"], "test")
        self.assertEqual(m1["level"], 1)
        self.assertEqual(m1["page"], 1)

        m2 = Metadata(header="test2", section="test2", level=1)
        self.assertEqual(m2.section, "test2")
        self.assertEqual(m2.header, "test2")
        self.assertEqual(m2.level, 1)
        self.assertEqual(m2.page, None)

    def test_missing_fields_metadata(self):
        # Test that missing header throws Validation Error
        with self.assertRaises(ValidationError):
            m = Metadata(section="test", level=1)

    def test_invalid_fields_metadata(self):
        # Test that level is int
        with self.assertRaises(ValidationError):
            m = Metadata(header="test", section="test", level="test")
        # Test that page is int
        with self.assertRaises(ValidationError):
            m = Metadata(header=3, section="test", level=1, page="test")


class TestChunk(unittest.TestCase):
    def test_chunk(self):
        c = Chunk(
            metadata=Metadata(section="test", header="test", level=1, page=1),
            content="test",
        ).model_dump()
        self.assertEqual(c["metadata"]["section"], "test")
        self.assertEqual(c["metadata"]["header"], "test")
        self.assertEqual(c["metadata"]["level"], 1)
        self.assertEqual(c["metadata"]["page"], 1)
        self.assertEqual(c["content"], "test")

    def test_missing_fields_chunk(self):
        # Test that missing content throws Validation Error
        with self.assertRaises(ValidationError):
            c = Chunk(metadata=Metadata(section="test", header="test", level=1))
        # Test that missing metadata is valid
        c = Chunk(content="test")
        self.assertEqual(c.metadata, None)
        self.assertEqual(c.content, "test")

    def test_invalid_fields_chunk(self):
        # Test that content is str
        with self.assertRaises(ValidationError):
            c = Chunk(content=1)
        # Test that metadata is Metadata
        with self.assertRaises(ValidationError):
            c = Chunk(metadata="test", content="test")
        # Test that Chunk with no metadata has type None
        c = Chunk(content="test")
        self.assertEqual(c.metadata, None)

    def test_json_init(self):
        c = Chunk(
            **{
                "metadata": {
                    "section": "section_test",
                    "header": "header_test",
                    "level": 1,
                    "page": 12,
                },
                "content": "content_test",
            }
        )
        self.assertEqual(type(c.metadata), Metadata)
        self.assertEqual(c.metadata.section, "section_test")
        self.assertEqual(c.metadata.header, "header_test")
        self.assertEqual(c.metadata.level, 1)
        self.assertEqual(c.metadata.page, 12)
        self.assertEqual(c.content, "content_test")


if __name__ == "__main__":
    unittest.main()
