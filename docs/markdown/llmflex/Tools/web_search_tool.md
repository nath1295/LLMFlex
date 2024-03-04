Module llmflex.Tools.web_search_tool
====================================

Functions
---------

    
`ddg_search(query: str, n: int = 5, urls_only: bool = True, **kwargs) ‑> List[Union[str, Dict[str, Any]]]`
:   Search with DuckDuckGo.
    
    Args:
        query (str): Search query.
        n (int, optional): Maximum number of results. Defaults to 5.
        urls_only (bool, optional): Only return the list of urls or return other information as well. Defaults to True.
    
    Returns:
        List[Union[str, Dict[str, Any]]]: List of search results.

Classes
-------

`WebSearchTool(embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit], name: str = 'web_search', description: str = 'This tool is for doing searches on the internet for facts or most updated information via a search engine.\nInput of this tool should be a search query or your question. \nOutput of this tool is the answer of your input question.', key_phrases: List[str] = ['use the browser', 'check online', 'search the internet'], search_engine: Literal['duckduckgo'] = 'duckduckgo', verbose: bool = True)`
:   This is the tool class for doing web search.
        
    
    Initialise the web search tool.
    
    Args:
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings to use for creating template
        name (str, optional): Name of the tool. Defaults to 'web_search'.
        description (str, optional): Description of the tool. Defaults to WEB_SEARCH_TOOL_DESCRIPTION.
        key_phrases (List[str], optional): List of key phrases to trigger the tool in the chat setup. Defaults to ['use the browser', 'check online', 'search the internet'].
        search_engine (Literal[&#39;duckduckgo&#39;], optional): Name of the search engine of the tool. Defaults to 'duckduckgo'.
        verbose: Whether to print logs while running the tool. Defaults to True.

    ### Ancestors (in MRO)

    * llmflex.Tools.base_tool.BaseTool
    * abc.ABC

    ### Methods

    `create_relevant_content_chunks(self, query: str, vectordb: llmflex.VectorDBs.faiss_vectordb.FaissVectorDatabase) ‑> Tuple[List[Dict[str, Any]], str]`
    :   Return the relevant chunks of contents from the vector database.
        
        Args:
            query (str): Search query.
            vectordb (FaissVectorDatabase): Vector database of search result contents.
        
        Returns:
            Tuple[List[Dict[str, Any]], str]: List of relevant chunks of contents and their links.

    `create_search_query(self, tool_input: str, llm: Optional[Type[llmflex.Models.Cores.base_core.BaseLLM]] = None, history: Union[List[str], List[Tuple[str, str]], ForwardRef(None)] = None, prompt_template: Optional[llmflex.Prompts.prompt_template.PromptTemplate] = None) ‑> str`
    :   Creating the search query for the search engine given the user input.
        
        Args:
            tool_input (str): User input.
            llm (Optional[Type[BaseLLM]], optional): LLM to create the search query. If not given, the search query will be the user input. Defaults to None.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): Recent conversation history as extra context for creating search query. Defaults to None.
            prompt_template (Optional[PromptTemplate], optional): Prompt template for structuring the prompt to create search query. Defaults to None.
        
        Returns:
            str: Search query.

    `create_vectordb(self, results: List[Dict[str, Any]], llm: Optional[Type[llmflex.Models.Cores.base_core.BaseLLM]] = None) ‑> llmflex.VectorDBs.faiss_vectordb.FaissVectorDatabase`
    :   Creating a temporary vector database of the search result contents.
        
        Args:
            results (List[Dict[str, Any]]): Search results from the search engine.
            llm (Optional[Type[BaseLLM]], optional): LLM for counting tokens to split contents. If none is given, the embeddings toolkit text splitter will be used. Defaults to None.
        
        Returns:
            FaissVectorDatabase: The temporary vector database.

    `generate_response(self, tool_input: str, chunks: List[Dict[str, Any]], llm: Type[llmflex.Models.Cores.base_core.BaseLLM], history: Union[List[str], List[Tuple[str, str]], ForwardRef(None)] = None, stream: bool = False, prompt_template: Optional[llmflex.Prompts.prompt_template.PromptTemplate] = None) ‑> Union[str, Iterator[str]]`
    :

    `search(self, query: str, n: int = 5, urls_only: bool = False, **kwargs) ‑> List[Union[str, Dict[str, Any]]]`
    :   Search with the given query.
        
        Args:
            query (str): Search query.
            n (int, optional): Maximum number of results. Defaults to 5.
            urls_only (bool, optional): Only return the list of urls or return other information as well. Defaults to True.
        
        Returns:
            List[Union[str, Dict[str, Any]]]: List of search results.