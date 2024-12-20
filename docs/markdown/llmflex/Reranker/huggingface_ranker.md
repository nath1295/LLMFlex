Module llmflex.Reranker.huggingface_ranker
==========================================

Classes
-------

`HuggingFaceRanker(pretrained_model_name_or_path: str = 'cross-encoder/ms-marco-TinyBERT-L-2-v2', model_kwargs: Dict[str, Any] | None = None, tokenizer_kwargs: Dict[str, Any] | None = None)`
:   A ranker that uses the HuggingFace transformers library for ranking documents.
    
    This ranker uses a pre-trained transformer model from the HugginFace transformers library to score and rank documents.
    
    Initializes the HuggingFaceRanker with a pretrained model and optional model and tokenizer kwargs.
    
    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model to use. Defaults to "cross-encoder/ms-marco-TinyBERT-L-2-v2".
        model_kwargs (Optional[Dict[str, Any]], optional): Optional keyword arguments to pass to the model. Defaults to None.
        tokenizer_kwargs (Optional[Dict[str, Any]], optional): Optional keyword arguments to pass to the tokenizer. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Reranker.base_ranker.BaseRanker
    * abc.ABC

    ### Methods

    `rerank(self, query: str, docs: List[str | llmflex.Schema.document.Document], top_k: int | None = None) ‑> List[llmflex.Reranker.base_ranker.RankResult]`
    :   Reranks the given list of documents based on the provided query using the model associated with this ranker.
        
        Args:
            query (str): The query string to use for reranking.
            docs (List[Union[str, Document]]): The list of documents to rerank. Each document can be either a string or a Document object.
            top_k (Optional[int], optional): The maximum number of documents to return. If not provided, all documents will be returned. Defaults to None.
            
        Returns:
            List[RankResult]: A list of RankResult tuples, where each tuple contains a Document and its corresponding score.