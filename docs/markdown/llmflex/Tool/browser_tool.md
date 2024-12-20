Module llmflex.Tool.browser_tool
================================

Functions
---------

`url_to_markdown(url: str, if_failed: Literal['raise', 'warning'] = 'warning') ‑> str`
:   Converts a URL to its Markdown representation.
    
    Args:
        url (str): The URL to convert.
        if_failed (Literal['raise', 'warning'], optional): The action to take if the URL cannot be converted. Defaults to 'warning'.
    
    Returns:
        str: The Markdown representation of the URL.

Classes
-------

`BrowserTool(base_url: str, embeddings: llmflex.Embeddings.Model.base_embeddings.BaseEmbeddings, tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer | None = None, ranker: llmflex.Reranker.base_ranker.BaseRanker | None = None, chunk_size: int = 300)`
:   A web search tool that uses a query or a list of queries provided by the user to perform web search via the SearXNG meta search engine API.
        
    
    Initializes the BrowserTool with the given parameters.
    
    Args:
        base_url (str): The base URL of the SearXNG search engine.
        embeddings (BaseEmbeddings): The embeddings model to use for relevant content searching in the search results.
        tokenizer (Optional[BaseTokenizer], optional): The tokenizer to use for token counting for result chunks. If None is provided, the tokenizer from the embedding model will be used. Defaults to None.
        ranker (Optional[BaseRanker], optional): The ranker to use for reranking the search results. If None is given, it will be initialised with ms-marco-TinyBERT. Defaults to None.
        chunk_size (int, optional): The maximum number of tokens in each chunk of the search results. Defaults to 300.

    ### Ancestors (in MRO)

    * llmflex.Tool.base_tool.BaseTool
    * abc.ABC

    ### Methods

    `run(self, **kwargs) ‑> llmflex.Tool.base_tool.ToolOutput`
    :   Run the tool and structure the output into a ToolOutput object.
        
        Returns:
            ToolOutput: Structured tool output.