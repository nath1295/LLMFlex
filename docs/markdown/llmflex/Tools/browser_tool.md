Module llmflex.Tools.browser_tool
=================================

Classes
-------

`BrowserTool(embeddings: llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit, llm: Optional[llmflex.Models.Cores.base_core.BaseLLM] = None, ranker: Optional[llmflex.Rankers.base_ranker.BaseRanker] = None)`
:   Tool for browsing contents via the DuckDuckGo search engine given any search query. The output will be the most relevant chunks of content found from the search engine according to the search query.
        
    
    Initialising the tool.
    
    Args:
        embeddings (BaseEmbeddingsToolkit): Embeddings toolkit for the vector database.
        llm (Optional[BaseLLM], optional): LLM to count number of tokens for each chunk. Defaults to None.
        ranker (Optional[BaseRanker], optional): Reranker to rerank results. If none is given, results will not be reranked after the search on the vector database. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Tools.tool_utils.BaseTool