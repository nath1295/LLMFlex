from pydantic import BaseModel
from typing import Dict, Any, Optional

class Document(BaseModel):
    index: str
    metadata: Dict[str, Any] = dict()

class RankResult:
    """Result of rerankers.
    """
    def __init__(self, index: str, rank_score: float, metadata: Optional[Dict[str, Any]] = None, original_score: Optional[float] = None, id: Optional[int] = None) -> None:
        from ..utils import validate_type
        self.index = validate_type(index, str)
        self.rank_score = validate_type(rank_score, float)
        self.metadata = dict() if metadata is None else validate_type(metadata, dict)
        self.original_score = 0.0 if original_score is None else validate_type(original_score, float)
        self.id = -1 if id is None else validate_type(id, int)

    def to_dict(self) -> Dict[str, Any]:
        """Transform the result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of the content of the result.
        """
        values = dict(
            index=self.index,
            rank_score=self.rank_score,
            metadata=self.metadata,
            original_score=self.original_score,
            id=self.id 
        )
        return values
    
    def __repr__(self) -> str:
        view = f'RankResult(index="{self.index}", rank_score={self.rank_score}, metadata={self.metadata}, original_score={self.original_score}, id={self.id})'
        return view