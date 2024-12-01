from abc import ABC, abstractmethod
from ..Schema.document import Document
from typing import Optional, List, Union, NamedTuple

class RankResult(NamedTuple):
    doc: Document
    score: float

class BaseRanker(ABC):
    """BaseRanker is an abstract base class for ranking documents using a given model.
    """
    def __init__(self, model_id: str) -> None:
        """Initializes a BaseRanker instance with the given model ID.

        Args:
            model_id (str): The ID of the model to use for ranking.
        """
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        """Gets the ID of the model used for ranking.

        Returns:
            str: The ID of the model.
        """
        return self._model_id
    
    @abstractmethod
    def rerank(self, query: str, docs: List[Union[str, Document]], top_k: Optional[int] = None) -> List[RankResult]:
        """Reranks the given list of documents based on the provided query using the model associated with this ranker.

        Args:
            query (str): The query string to use for reranking.
            docs (List[Union[str, Document]]): The list of documents to rerank. Each document can be either a string or a Document object.
            top_k (Optional[int], optional): The maximum number of documents to return. If not provided, all documents will be returned. Defaults to None.

        Returns:
            List[RankResult]: A list of RankResult tuples, where each tuple contains a Document and its corresponding score.
        """
        pass