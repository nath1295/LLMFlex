from .base_ranker import BaseRanker
from ..Schemas.documents import Document, RankResult
from typing import List, Dict, Union, Any

class FlashrankRanker(BaseRanker):
    """Class for FlashRank rerankers.
    """
    def __init__(self, model_name: str =  'ms-marco-TinyBERT-L-2-v2', max_length: int = 512) -> None:
        """Initialise the ranker.

        Args:
            model_name (str, optional): Model to use for reranking. Please check https://github.com/PrithivirajDamodaran/FlashRank for more details. Defaults to 'ms-marco-TinyBERT-L-2-v2'.
            max_length (int, optional): Maximum number of tokens per document. Defaults to 512.
        """
        from flashrank import Ranker
        from ..utils import get_config
        import os
        super().__init__()
        cache_dir = os.path.join(get_config()['hf_home'], 'flashrank')
        os.makedirs(cache_dir, exist_ok=True)
        self._model = Ranker(model_name=model_name, max_length=max_length, cache_dir=cache_dir)

    def rerank(self, query: str, elements: List[Union[Document, Dict[str, Any]]], top_k: int = 5) -> List[RankResult]:
        """The method to rerank list of documents, usually from the search results of a vector database.

        Args:
            query (str): Query for reranking the given list of documents.
            elements (List[RankResult]): List of documents or dictionaries of search results to rerank.
            top_k (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of results ordered descendingly.
        """
        from flashrank import RerankRequest
        num_docs = len(elements)
        docs = map(self._format_element, elements)
        docs = list(map(lambda x: dict(id=x['id'], text=x['text'], meta=dict(metadata=x['metadata'], original_score=x['original_score'])), docs))
        request = RerankRequest(query=query, passages=docs)
        results = self._model.rerank(request=request)
        results = results if len(results) <= top_k else results[:top_k]
        results = list(map(lambda x: RankResult(index=x['text'], metadata=x['meta']['metadata'], 
                    rank_score=float(x['score']), original_score=float(x['meta']['original_score']), id=x['id']), results))
        return results


