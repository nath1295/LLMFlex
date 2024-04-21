Module llmflex.Rankers.flashrank_ranker
=======================================

Classes
-------

`FlashrankRanker(model_name: str = 'ms-marco-TinyBERT-L-2-v2', max_length: int = 512)`
:   Class for FlashRank rerankers.
        
    
    Initialise the ranker.
    
    Args:
        model_name (str, optional): Model to use for reranking. Please check https://github.com/PrithivirajDamodaran/FlashRank for more details. Defaults to 'ms-marco-TinyBERT-L-2-v2'.
        max_length (int, optional): Maximum number of tokens per document. Defaults to 512.

    ### Ancestors (in MRO)

    * llmflex.Rankers.base_ranker.BaseRanker
    * abc.ABC