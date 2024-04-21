Module llmflex.Rankers.base_ranker
==================================

Classes
-------

`BaseRanker()`
:   Base class for rerankers.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.Rankers.flashrank_ranker.FlashrankRanker

    ### Methods

    `rerank(self, query: str, elements: List[Union[llmflex.Schemas.documents.Document, Dict[str, Any]]], top_k: int = 5) ‑> List[llmflex.Schemas.documents.RankResult]`
    :   The method to rerank list of documents, usually from the search results of a vector database.
        
        Args:
            query (str): Query for reranking the given list of documents.
            elements (List[RankResult]): List of documents or dictionaries of search results to rerank.
            top_k (int, optional): Maximum number of results to return. Defaults to 5.
        
        Returns:
            List[Dict[str, Any]]: List of results ordered descendingly.