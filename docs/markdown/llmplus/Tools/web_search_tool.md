Module llmplus.Tools.web_search_tool
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

    
`parse_url(url: str) ‑> str`
:   Parse the given URL as markdown.
    
    Args:
        url (str): URL to parse.
    
    Returns:
        str: Content of the URL as markdown.

Classes
-------

`WebSearchTool(embeddings: llmplus.Embeddings.base_embeddings.BaseEmbeddingsToolkit, name: str = 'web_search', description: str = 'This tool is for doing searches on the internet for facts or most updated information via a search engine.\nInput of this tool should be a search query. \nOutput of this tool is the answer of your input question.', search_engine: Literal['duckduckgo'] = 'duckduckgo')`
:   This is the tool class for doing web search.
        
    
    Initialise teh web search tool.
    
    Args:
        embeddings (BaseEmbeddingsToolkit): Embeddings to use for creating template
        name (str, optional): Name of the tool. Defaults to 'web_search'.
        description (str, optional): Description of the tool. Defaults to WEB_SEARCH_TOOL_DESCRIPTION.
        search_engine (Literal[&#39;duckduckgo&#39;], optional): Name of the search engine of the tool. Defaults to 'duckduckgo'.

    ### Ancestors (in MRO)

    * llmplus.Tools.base_tool.BaseTool

    ### Methods

    `run(self, tool_input: str, llm: llmplus.Models.Cores.base_core.BaseLLM = None, stream: bool = False, history: Optional[List[List[str]]] = None, prompt_template: Optional[llmplus.Prompts.prompt_template.PromptTemplate] = None, generate_query: bool = True, return_type: Literal['response', 'vectordb', 'chunks'] = 'response', **kwargs) ‑> Union[str, Iterator[str], List[Dict[str, Any]], Any]`
    :   Run the web search tool. Any keyword arguments will be passed to the search method.
        
        Args:
            tool_input (str): Input of the tool, usually the latest user input in the chatbot conversation.
            llm (BaseLLM, optional): It will be used to create the search query and generate output if `generate_query=True`. 
            stream (bool, optional): If an llm is provided and `stream=True`, A generator of the output will be returned. Defaults to False.
            history (Optional[List[List[str]]], optional): Snippet of recent chat history to help forming more relevant search if provided. Defaults to None.
            prompt_template (Optional[PromptTemplate], optional): Prompt template use to format the chat history. Defaults to None.
            generate_query (bool, optional): Whether to treat the tool_input as part of the conversation and generate a different search query. Defaults to True.
            return_type (Literal['response', 'vectordb', 'chunks'], optional): Return a full response given the tool_input, the vector database, or the chunks only. Defaults to 'response'.
        
        Returns:
            Union[str, Iterator[str], List[Dict[str, Any]], Any]: Search result, if llm and prompt template is provided, the result will be provided as a reponse to the tool_input.

    `search(self, query: str, n: int = 5, urls_only: bool = True, **kwargs) ‑> List[Union[str, Dict[str, Any]]]`
    :   Search with the given query.
        
        Args:
            query (str): Search query.
            n (int, optional): Maximum number of results. Defaults to 5.
            urls_only (bool, optional): Only return the list of urls or return other information as well. Defaults to True.
        
        Returns:
            List[Union[str, Dict[str, Any]]]: List of search results.