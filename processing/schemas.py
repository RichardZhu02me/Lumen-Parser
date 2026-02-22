from typing import Optional, List
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    section: Optional[str] = Field(None, description="Section header")
    header: str = Field(description="Header text")
    level: Optional[int] = Field(None, description="Header level")
    page: Optional[int] = Field(None, description="Page Number")


class Chunk(BaseModel):
    metadata: Optional[Metadata] = Field(None, description="Metadata")
    content: str = Field(..., description="Content")


class HeaderModification(BaseModel):
    header_name: str = Field(description="Header Name")
    modified_level: int = Field(
        description="Proposed Header level to change the header to"
    )
    reason: Optional[str] = Field(description="Reason for modification")


class DocumentAnalysis(BaseModel):
    """Analysis of header corrections"""

    structure_style: str = Field(description="Proposed Structure style of the Document")
    header_modifications: List[HeaderModification] = Field(
        description="List of proposed modifications to headers"
    )
