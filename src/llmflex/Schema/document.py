from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Document(BaseModel):
    """Class for texts with metadata.
    """
    text: str = Field(description='Text.')
    metadata: Dict[str, Any] = Field(description='Metadata of the document.', default_factory=dict)
