Module llmflex.Reranker.base_ranker
===================================

Classes
-------

`BaseRanker(model_id: str)`
:   BaseRanker is an abstract base class for ranking documents using a given model.
        
    
    Initializes a BaseRanker instance with the given model ID.
    
    Args:
        model_id (str): The ID of the model to use for ranking.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.Reranker.huggingface_ranker.HuggingFaceRanker

    ### Instance variables

    `model_id: str`
    :   Gets the ID of the model used for ranking.
        
        Returns:
            str: The ID of the model.

    ### Methods

    `rerank(self, query: str, docs: List[str | llmflex.Schema.document.Document], top_k: int | None = None) ‑> List[llmflex.Reranker.base_ranker.RankResult]`
    :   Reranks the given list of documents based on the provided query using the model associated with this ranker.
        
        Args:
            query (str): The query string to use for reranking.
            docs (List[Union[str, Document]]): The list of documents to rerank. Each document can be either a string or a Document object.
            top_k (Optional[int], optional): The maximum number of documents to return. If not provided, all documents will be returned. Defaults to None.
        
        Returns:
            List[RankResult]: A list of RankResult tuples, where each tuple contains a Document and its corresponding score.

`RankResult(doc: llmflex.Schema.document.Document, score: float)`
:   RankResult(doc, score)

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `doc: llmflex.Schema.document.Document`
    :   Alias for field number 0

    `score: float`
    :   Alias for field number 1