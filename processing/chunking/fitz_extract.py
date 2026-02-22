import pymupdf.layout
import pymupdf4llm
import pathlib

# doc = pymupdf.open("data/raw/doc1.pdf")
md = pymupdf4llm.to_markdown("data/raw/resume.pdf")
pathlib.Path("data/extracted/output.md").write_bytes(md.encode())


