from pydantic import BaseModel
from typing import Dict, Any

class Document(BaseModel):
    index: str
    metadata: Dict[str, Any] = dict()