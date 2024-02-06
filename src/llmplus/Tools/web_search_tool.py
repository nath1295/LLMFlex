from ..Models.Cores.base_core import BaseLLM
from ..Prompts.prompt_template import PromptTemplate
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..Data.vector_database import VectorDatabase
from .base_tool import BaseTool
from typing import Iterator, List, Dict, Any, Optional, Union, Literal, Type, Tuple

WEB_SEARCH_TOOL_DESCRIPTION = """This tool is for doing searches on the internet for facts or most updated information via a search engine.
Input of this tool should be a search query or your question. 
Output of this tool is the answer of your input question."""

QUERY_GENERATION_SYS_RPOMPT = """You are an AI assistant who is analysing the conversation you are having with the user. You need to use a search engine to search for the most relevant information that can help you to give the user the most accurate and coherent respond. The user is asking you to generate the most appropriate search query for the latest user request.

Here are the most recent conversations you have with the user:
"""

SEARCH_RESPONSE_SYS_RPOMPT = """You are a helpful AI assistant having a conversation with a user. You have just used a search engine to get some relevant information that might help you to respond to the user's latest request. Here are some relevant chunks of contents that you found with the search engine. Use them to respond to the users if they are useful.

Relevant chunks of contents:

"""

def ddg_search(query: str, n: int = 5, urls_only: bool = True, **kwargs) -> List[Union[str, Dict[str, Any]]]:
    """Search with DuckDuckGo.

    Args:
        query (str): Search query.
        n (int, optional): Maximum number of results. Defaults to 5.
        urls_only (bool, optional): Only return the list of urls or return other information as well. Defaults to True.

    Returns:
        List[Union[str, Dict[str, Any]]]: List of search results.
    """
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=n, **kwargs)]
    if urls_only:
        results = list(map(lambda x: x['href'], results))
    return results
    
class WebSearchTool(BaseTool):
    """This is the tool class for doing web search.
    """
    def __init__(self, embeddings: Type[BaseEmbeddingsToolkit], 
                 name: str = 'web_search', description: str = WEB_SEARCH_TOOL_DESCRIPTION, 
                 search_engine: Literal['duckduckgo'] = 'duckduckgo', verbose: bool = True) -> None:
        """Initialise the web search tool.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings to use for creating template
            name (str, optional): Name of the tool. Defaults to 'web_search'.
            description (str, optional): Description of the tool. Defaults to WEB_SEARCH_TOOL_DESCRIPTION.
            search_engine (Literal[&#39;duckduckgo&#39;], optional): Name of the search engine of the tool. Defaults to 'duckduckgo'.
            verbose: Whether to print logs while running the tool. Defaults to True.
        """
        super().__init__(name, description, verbose)
        self.search_engine = search_engine
        self.embeddings = embeddings

    def create_search_query(self, tool_input: str, 
            llm: Optional[Type[BaseLLM]] = None, 
            history: Optional[Union[List[str], List[Tuple[str, str]]]] = None, 
            prompt_template: Optional[PromptTemplate] = None) -> str:
        """Creating the search query for the search engine given the user input.

        Args:
            tool_input (str): User input.
            llm (Optional[Type[BaseLLM]], optional): LLM to create the search query. If not given, the search query will be the user input. Defaults to None.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): Recent conversation history as extra context for creating search query. Defaults to None.
            prompt_template (Optional[PromptTemplate], optional): Prompt template for structuring the prompt to create search query. Defaults to None.

        Returns:
            str: Search query.
        """
        tool_input = tool_input.strip(' \n\r\t')
        generate_query = True if llm is not None else False
        if not generate_query:
            query = tool_input
            self.print(f'Search query: "{query}"')
        else:
            if ((history is not None) & (prompt_template is not None)):
                conversation = prompt_template.format_history(history=history) + prompt_template.human_prefix + tool_input
            else:
                prompt_template = llm.core.prompt_template if prompt_template is None else prompt_template
                if history is not None:
                    conversation = prompt_template.format_history(history=history) + prompt_template.human_prefix + tool_input
                else:
                    conversation = prompt_template.human_prefix + tool_input
            request = f'This is my latest request: {tool_input}\n\nGenerate the search query that helps you to search in the search engine and respond, in JSON format.'
            query_prompt = prompt_template.create_prompt(user=request, system=QUERY_GENERATION_SYS_RPOMPT + conversation)
            query_prompt += '```json\n{"Search query": "'
            query = '{"Search query": "' + llm.invoke(query_prompt, stop=['```'], temperature=0)
            query = query.rstrip('`')
            try:
                import json
                query = json.loads(query)['Search query']
            except:
                self.print(f'Generation of query failed, fall back to use the raw tool_input "{tool_input}".')
                query = tool_input
        return query

    def search(self, query: str, n: int = 5, urls_only: bool = False, **kwargs) -> List[Union[str, Dict[str, Any]]]:
        """Search with the given query.

        Args:
            query (str): Search query.
            n (int, optional): Maximum number of results. Defaults to 5.
            urls_only (bool, optional): Only return the list of urls or return other information as well. Defaults to True.

        Returns:
            List[Union[str, Dict[str, Any]]]: List of search results.
        """
        if self.search_engine == 'duckduckgo':
            return ddg_search(query=query, n=n, urls_only=urls_only, **kwargs)
        else:
            raise ValueError(f'Search engine "{self.search_engine}" not supported.')
        
    def create_vectordb(self, results: List[Dict[str, Any]], llm: Optional[Type[BaseLLM]] = None) -> VectorDatabase:
        """Creating a temporary vector database of the search result contents.

        Args:
            results (List[Dict[str, Any]]): Search results from the search engine.
            llm (Optional[Type[BaseLLM]], optional): LLM for counting tokens to split contents. If none is given, the embeddings toolkit text splitter will be used. Defaults to None.

        Returns:
            VectorDatabase: The temporary vector database.
        """
        from .web_search_utils import get_markdown, create_content_chunks
        from ..TextSplitters.llm_text_splitter import LLMTextSplitter
        from langchain.schema.document import Document

        urls = list(map(lambda x: x['href'], results))
        vectordb = VectorDatabase.from_empty(embeddings=self.embeddings)

        if llm is None:
            contents = list(map(lambda x: get_markdown(x, as_list=False), urls))
            docs = list(map(lambda x: Document(page_content=x[0], metadata=x[1]), list(zip(contents, results))))
            vectordb.add_documents(docs=docs, text_splitter=self.embeddings.text_splitter, split_text=True)
        else:
            contents = list(map(lambda x: get_markdown(x, as_list=True), urls))
            contents = list(map(lambda x: create_content_chunks(x, llm), contents))
            docs = list(zip(contents, results))
            docs = list(map(lambda x: list(map(lambda y: Document(page_content=y, metadata=x[1]), x[0])), docs))
            docs = sum(docs, [])
            vectordb.add_documents(docs=docs, split_text=False)
        return vectordb
    
    def create_relevant_content_chunks(self, query: str, vectordb: VectorDatabase) -> Tuple[List[Dict[str, Any]], str]:
        """Return the relevant chunks of contents from the vector database.

        Args:
            query (str): Search query.
            vectordb (VectorDatabase): Vector database of search result contents.

        Returns:
            Tuple[List[Dict[str, Any]], str]: List of relevant chunks of contents and their links.
        """
        chunks = vectordb.search(query=query, top_k=3, index_only=False)
        links = list(set(list(map(lambda x: x['metadata']['href'], chunks))))
        link_str = []
        for i, link in enumerate(links):
            link_str.append(f'{i + 1}. {link}  ')
        links = 'Sources:  \n' + '\n'.join(link_str)
        return chunks, links
    
    def generate_response(self, tool_input: str,  
                          chunks: List[Dict[str, Any]], 
                          llm: Type[BaseLLM],
                          history: Optional[Union[List[str], List[Tuple[str, str]]]] = None, 
                          stream: bool = False, 
                          prompt_template: Optional[PromptTemplate] = None) -> Union[str, Iterator[str]]:
        
        rel_info = list(map(lambda x: x['index'], chunks))
        rel_info = '\n\n'.join(rel_info) + '\n'

        prompt_template = llm.core.prompt_template if prompt_template is None else prompt_template
        prompt = prompt_template.create_prompt(user=tool_input, system=SEARCH_RESPONSE_SYS_RPOMPT + rel_info, history=history if history is not None else [])
        from ..Models.Cores.utils import add_newline_char_to_stopwords
        stop = add_newline_char_to_stopwords(prompt_template.stop)
        if llm is None:
            raise ValueError(f'A llm has to be provided to generate response.')
        if stream:
            return llm.stream(prompt, stop=stop)
        else:
            return llm.invoke(prompt, stop=stop)

    def _tool_schema(self) -> Dict[str, Any]:
        schema = {
            'create_search_query' : dict(
                input=['tool_input', 'llm', 'history', 'prompt_template'],
                output=['query']
            ),
            'search' : dict(
                input=['query'],
                output=['results']
            ),
            'create_vectordb' : dict(
                input=['results', 'llm'],
                output=['vectordb']
            ),
            'create_relevant_content_chunks' : dict(
                input=['query', 'vectordb'],
                output=['chunks', 'extra_info']
            ),
            'generate_response' : dict(
                input=['tool_input', 'chunks', 'llm', 'history', 'stream', 'prompt_template'],
                output=['final_output']
            )
        }
        return schema