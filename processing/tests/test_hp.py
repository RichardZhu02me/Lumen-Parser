import sys
import os

# add backend to python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import unittest
from unittest.mock import patch
import json
from processing.hierarchy_parsing import (
    get_header_chunks,
    modify_header_level,
    correct_headers,
    parse_sections,
    output_chunks,
    output_chunks_to_file,
)
from processing.schemas import DocumentAnalysis, Chunk, Metadata
from processing.tests.mock_dc import mock_doc_analysis


class TestHeaderChunks(unittest.TestCase):
    def test_get_header_chunks(self):
        with open("data/extracted/test/resume.md", "r") as f:
            text = f.read()
        header_chunks = get_header_chunks(text)
        self.assertEqual(len(header_chunks), 13)

    def test_modify_header_level(self):
        with open("data/extracted/test/resume.md", "r") as f:
            text = f.read()
        header_chunks = get_header_chunks(text)
        content = header_chunks[0].content
        header_name = header_chunks[0].metadata.header
        modify_header_level(header_chunks[0], 1)
        self.assertEqual(header_chunks[0].metadata.level, 1)
        self.assertEqual(header_chunks[0].content, content)
        self.assertEqual(header_chunks[0].metadata.header, header_name)

    @patch("ml.agent.Agent.invoke", autospec=True)
    def test_correct_headers_mock(self, mock_chatcompletion):
        mock_doc_analysis = DocumentAnalysis(
            **{
                "header_modifications": [
                    {
                        "header_name": "Header 2",
                        "modified_level": 1,
                        "reason": "reason",
                    },
                    {
                        "header_name": "Link",
                        "modified_level": 3,
                        "reason": "there is no reason",
                    },
                ],
                "structure_style": "test",
            }
        )
        mock_response = json.dumps(mock_doc_analysis.model_dump())

        mock_chatcompletion.return_value = mock_response
        with open("data/extracted/test/sample_doc.md", "r") as f:
            text = f.read()
        header_chunks = get_header_chunks(text)
        header_cache = []
        for i in range(8):
            header_cache.append(header_chunks[i].metadata.level)
        self.assertEqual(len(header_chunks), 9)
        self.assertEqual(header_chunks[1].metadata.level, 2)
        self.assertEqual(header_chunks[8].metadata.level, 6)
        correct_headers(header_chunks)
        self.assertEqual(header_chunks[1].metadata.level, 1)
        self.assertEqual(header_chunks[8].metadata.level, 3)
        for i in range(8):
            if i == 1 or i == 8:
                continue
            self.assertEqual(header_chunks[i].metadata.level, header_cache[i])
        mock_chatcompletion.assert_called_once()

    # def test_correct_headers(self):
    #     with open("data/extracted/test/resume.md", "r") as f:
    #         text = f.read()
    #     header_chunks = get_header_chunks(text)
    #     correct_headers(header_chunks)
    #     for header in header_chunks:
    #         print(header)

    @patch("ml.agent.Agent.invoke", autospec=True)
    def test_parse_sections(self, mock_chatcompletion):
        mock_response = json.dumps(mock_doc_analysis.model_dump())
        mock_chatcompletion.return_value = mock_response
        with open("data/extracted/test/resume.md", "r") as f:
            text = f.read()
        child_chunks, parent_chunks = parse_sections(text)
        self.assertEqual(len(child_chunks), 37)
        self.assertEqual(len(parent_chunks), 5)
        for chunk in child_chunks:
            assert chunk.metadata.section is not None
        assert parent_chunks[0].metadata.section == "**Richard Zhu**"
        assert parent_chunks[1].metadata.section == "Experience"
        assert parent_chunks[2].metadata.section == "Projects"
        assert parent_chunks[3].metadata.section == "Technical Skills"
        assert parent_chunks[4].metadata.section == "Education"

    @patch("ml.agent.Agent.invoke", autospec=True)
    def test_output_chunks_to_file(self, mock_chatcompletion):
        mock_response = json.dumps(mock_doc_analysis.model_dump())
        mock_chatcompletion.return_value = mock_response
        with open("data/extracted/test/resume.md", "r") as f:
            text = f.read()
        child_chunks, parent_chunks = parse_sections(text)
        output_chunks_to_file(
            child_chunks, "data/processed/test/resume_child_chunks.json"
        )
        output_chunks_to_file(
            parent_chunks, "data/processed/test/resume_parent_chunks.json"
        )


if __name__ == "__main__":
    unittest.main()
