Module llmflex.Tools.browser_tool
=================================

Classes
-------

`BrowserTool(embeddings: llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit, llm: Optional[llmflex.Models.Cores.base_core.BaseLLM] = None, ranker: Optional[llmflex.Rankers.base_ranker.BaseRanker] = None)`
:   Tool for browsing contents via the DuckDuckGo search engine given any search query. The output will be the most relevant chunks of content found from the search engine according to the search query.
        
    
    Initialising the tool.
    
    Args:
        name (Optional[str], optional): Name of the tool. If not given, it will be the tool class name. Defaults to None.
        description (Optional[str], optional): Description of the tool. If not given, it will read from the docstring of the tool class. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Tools.tool_utils.BaseTool