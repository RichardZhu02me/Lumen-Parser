import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from .schemas import Chunk, DocumentAnalysis
from ml import agent
from .templates import get_header_correction_template
import json
from typing import Tuple, List


def get_header_level(header: str):
    return len(re.match(r"^#{1,6}", header).group(0))


def get_header_chunks(text: str) -> List[Chunk]:
    lines = text.split("\n")
    header_chunks = []
    for line in lines:
        if re.match(r"^#{1,6}", line):
            header_chunks.append(
                Chunk(
                    metadata={
                        "header": line.strip("#").strip(),
                        "level": get_header_level(line),
                    },
                    content=line,
                )
            )
    return header_chunks


def modify_header_level(header: Chunk, level: int):
    header.metadata.level = level


def correct_headers(headers: List[Chunk]) -> None:
    """Side Effects: Modifies the header level of chunks in place"""
    template = get_header_correction_template().render(headers=headers)
    response = agent.invoke(
        template,
        config={
            "response_mime_type": "application/json",
            "response_schema": DocumentAnalysis,
        },
    )
    modifications = DocumentAnalysis(**json.loads(response))
    ## create a dict for header references
    header_map = {}
    for header in headers:
        header_map[header.metadata.header] = header

    for mod in modifications.header_modifications:
        modify_header_level(
            header_map[mod.header_name],
            mod.modified_level,
        )
    return


""" Assumptions:
    - Size of Header Chunks is equal to number of headers
    - Headers are in order
"""


def parse_sections(text: str) -> Tuple[List[Chunk], List[Chunk]]:
    lines = text.split("\n")

    header_chunks = get_header_chunks(text)
    correct_headers(header_chunks)
    header_index = 0
    child_chunks = []
    parent_chunks = []
    parent_content = []
    header_stack = []

    def add_parent_chunk(section: Chunk):
        meta = section.metadata
        parent_chunks.append(
            Chunk(
                metadata={
                    "section": meta.header,
                    "header": meta.header,
                    "page": meta.page,
                },
                content="\n".join(parent_content),
            )
        )

    def add_child_chunk(line: str, header, section):
        metadata = {
            "header": header.metadata.header,
            "section": section.metadata.header,
            "page": section.metadata.page,
        }
        child_chunks.append(Chunk(**{"metadata": metadata, "content": line}))

    section = ""
    for line in lines:
        # skip empty lines
        if re.match(r"^[\n\s]*$", line):
            continue
        # if not a header, add to parent content and create child chunk
        if not re.match(r"^#{1,6} ", line):
            parent_content.append(line)
            add_child_chunk(line, header_stack[-1], header_stack[0])
            continue

        ## This is a header line

        ## adds a header to the stack if there is no stack
        if len(header_stack) == 0:
            header_stack.append(header_chunks[header_index])
            parent_content.append(line)
            add_child_chunk(line, header_stack[-1], header_stack[0])
            header_index += 1
            continue

        ## update header stack

        header_level = header_chunks[header_index].metadata.level

        print("header_level: ", header_level)

        section: Chunk = header_stack[0]
        while len(header_stack) > 0 and header_level <= header_stack[-1].metadata.level:
            print("popping header {}".format(header_stack[-1].metadata.header))
            header_stack.pop()
        if len(header_stack) == 0:
            add_parent_chunk(section)
            parent_content = []

        header_stack.append(header_chunks[header_index])
        add_child_chunk(line, header_stack[-1], header_stack[0])
        parent_content.append(line)
        header_index += 1

    add_parent_chunk(header_stack[0])
    return child_chunks, parent_chunks


def output_chunks(chunks: List[Chunk]):
    for chunk in chunks:
        print(chunk)


def output_chunks_to_file(chunks: List[Chunk], file_path: str):
    with open(file_path, "w") as f:
        f.write(json.dumps([chunk.model_dump() for chunk in chunks]))
