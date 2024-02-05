from ..Models.Cores.base_core import BaseLLM
from ..Prompts.prompt_template import PromptTemplate
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from .base_tool import BaseTool
from typing import Iterator, List, Dict, Any, Optional, Union, Literal, Type, Tuple

WEB_SEARCH_TOOL_DESCRIPTION = """This tool is for doing searches on the internet for facts or most updated information via a search engine.
Input of this tool should be a search query. 
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
        """Initialise teh web search tool.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings to use for creating template
            name (str, optional): Name of the tool. Defaults to 'web_search'.
            description (str, optional): Description of the tool. Defaults to WEB_SEARCH_TOOL_DESCRIPTION.
            search_engine (Literal[&#39;duckduckgo&#39;], optional): Name of the search engine of the tool. Defaults to 'duckduckgo'.
            verbose: Whether to print logs while running the tool. Defaults to True.
        """
        super().__init__(name, description, verbose)
        from ..Data.vector_database import VectorDatabase
        self.search_engine = search_engine
        self.embeddings = embeddings

    def search(self, query: str, n: int = 5, urls_only: bool = True, **kwargs) -> List[Union[str, Dict[str, Any]]]:
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
            return [f'Search engine "{self.search_engine}" not supported.']
        

    def run(self, tool_input: str, llm: Optional[Type[BaseLLM]] = None, stream: bool = False, 
            history: Optional[Union[List[str], List[Tuple[str, str]]]] = None, prompt_template: Optional[PromptTemplate] = None, 
            generate_query: bool = True, return_type: Literal['response', 'vectordb', 'chunks'] = 'response', **kwargs) -> Union[str, Iterator[str], List[Dict[str, Any]], Any]:
        """Run the web search tool. Any keyword arguments will be passed to the search method.

        Args:
            tool_input (str): Input of the tool, usually the latest user input in the chatbot conversation.
            llm (Optional[Type[BaseLLM]], optional): It will be used to create the search query and generate output if `generate_query=True`. 
            stream (bool, optional): If an llm is provided and `stream=True`, A generator of the output will be returned. Defaults to False.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): Snippet of recent chat history to help forming more relevant search if provided. Defaults to None.
            prompt_template (Optional[PromptTemplate], optional): Prompt template use to format the chat history. Defaults to None.
            generate_query (bool, optional): Whether to treat the tool_input as part of the conversation and generate a different search query. Defaults to True.
            return_type (Literal['response', 'vectordb', 'chunks'], optional): Return a full response given the tool_input, the vector database, or the chunks only. Defaults to 'response'.

        Returns:
            Union[str, Iterator[str], List[Dict[str, Any]], Any]: Search result, if llm and prompt template is provided, the result will be provided as a reponse to the tool_input.
        """
        tool_input = tool_input.strip(' \n\r\t')
        generate_query = generate_query if llm is not None else False
        if not generate_query:
            query = tool_input
            self.print(f'Search query: "{query}"')
        else:
            if ((history is not None) & (prompt_template is not None)):
                conversation = prompt_template.format_history(history=history) + prompt_template.human_prefix + tool_input
            elif prompt_template is not None:
                conversation = prompt_template.human_prefix + tool_input
            elif history is not None:
                raise ValueError('Prompt template need to be provided to process chat history.')
            else:
                conversation = 'User: ' + tool_input
            prompt_template = PromptTemplate.from_preset('Default Instruct') if prompt_template is None else prompt_template
            request = f'This is my latest request: {tool_input}\n\nGenerate the search query that helps you to search in the search engine and respond, in JSON format.'
            query_prompt = prompt_template.create_prompt(user=request, system=QUERY_GENERATION_SYS_RPOMPT + conversation)
            query_prompt += '```json\n{"Search query": "'
            query = '{"Search query": "' + llm(query_prompt, stop=['```'])
            query = query.rstrip('`')
            try:
                import json
                query = json.loads(query)['Search query']
                self.print(f'Search query: "{query}"')
            except:
                self.print(f'Generation of query failed, fall back to use the raw tool_input "{tool_input}".')
                query = tool_input

        from ..TextSplitters.llm_text_splitter import LLMTextSplitter
        from ..Models.Cores.utils import add_newline_char_to_stopwords
        from .web_search_utils import get_markdown, create_content_chunks
        from langchain.schema.document import Document
        from ..Data.vector_database import VectorDatabase

        text_splitter = LLMTextSplitter(model=llm)
        results = self.search(query=query, urls_only=False, **kwargs)
        urls = list(map(lambda x: x['href'], results))
        vectordb = VectorDatabase.from_empty(embeddings=self.embeddings)
        if llm is None:
            contents = list(map(lambda x: get_markdown(x, as_list=False), urls))
            self.print('Parsing contents completed.')
            docs = list(map(lambda x: Document(page_content=x[0], metadata=x[1]), list(zip(contents, results))))
            vectordb.add_documents(docs=docs, text_splitter=text_splitter, split_text=True)
            self.print('Storing contents completed.')
        else:
            contents = list(map(lambda x: get_markdown(x, as_list=True), urls))
            self.print('Parsing contents completed.')
            contents = list(map(lambda x: create_content_chunks(x, llm), contents))
            docs = list(zip(contents, results))
            docs = list(map(lambda x: list(map(lambda y: Document(page_content=y, metadata=x[1]), x[0])), docs))
            docs = sum(docs, [])
            vectordb.add_documents(docs=docs, split_text=False)
            self.print('Storing contents completed.')

        if return_type == 'vectordb':
            return vectordb
        
        chunks = vectordb.search(query=query, top_k=3, index_only=False)
        if return_type == 'chunks':
            return chunks
        
        rel_info = list(map(lambda x: x['index'], chunks))
        rel_info = '\n\n'.join(rel_info) + '\n'

        prompt = prompt_template.create_prompt(user=tool_input, system=SEARCH_RESPONSE_SYS_RPOMPT + rel_info, history=history if history is not None else [])
        stop = add_newline_char_to_stopwords(prompt_template.stop)
        if llm is None:
            raise ValueError(f'A llm has to be provided to generate response.')
        if stream:
            return llm.stream(prompt, stop=stop)
        else:
            return llm(prompt, stop=stop)
        
