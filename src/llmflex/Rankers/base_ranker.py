from abc import ABC, abstractmethod
from ..Schemas.documents import Document, RankResult
from typing import List, Dict, Union, Any

class BaseRanker(ABC):
    """Base class for rerankers.
    """

    @abstractmethod
    def rerank(self, query: str, elements: List[Union[Document, Dict[str, Any]]], top_k: int = 5) -> List[RankResult]:
        """The method to rerank list of documents, usually from the search results of a vector database.

        Args:
            query (str): Query for reranking the given list of documents.
            elements (List[RankResult]): List of documents or dictionaries of search results to rerank.
            top_k (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of results ordered descendingly.
        """
        pass

    def _format_element(self, element: Union[Document, Dict[str, Any]]) -> Dict[str, Any]:
        """Standardise the document for reranking.

        Args:
            element (Union[Document, Dict[str, Any]]): Document to standardise.

        Returns:
            Dict[str, Any]: Standardised dictionary of the document.
        """
        if isinstance(element, Document):
            doc = dict(text=element.index, metadata=element.metadata, original_score=0.0, id=-1)
        else:
            doc = dict(text=element['index'], metadata=element['metadata'], original_score=element['score'], id=element['id'])
        return doc