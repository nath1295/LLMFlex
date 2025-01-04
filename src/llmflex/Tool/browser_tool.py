from .base_tool import BaseTool, ToolOutput
from requests import get
from typing import Any, Literal, Optional, List, Dict, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from ..Embeddings.Model.base_embeddings import BaseEmbeddings
    from ..Tokenizer.base_tokenizer import BaseTokenizer
    from ..Reranker.base_ranker import BaseRanker

def url_to_markdown(url: str, if_failed: Literal['raise', 'warning'] = 'warning') -> str:
    """Converts a URL to its Markdown representation.

    Args:
        url (str): The URL to convert.
        if_failed (Literal['raise', 'warning'], optional): The action to take if the URL cannot be converted. Defaults to 'warning'.

    Returns:
        str: The Markdown representation of the URL.
    """
    try:
        from markdownify import markdownify
    except:
        raise ModuleNotFoundError(f'Module "markdownify" not installed. Please install it with `pip`.')
    try:
        r = get(url)
        r.raise_for_status()
    except Exception as e:
        if if_failed == 'raise':
            raise e
        else:
            import warnings
            warnings.warn(str(e))
            return ''
    html = r.text
    md = markdownify(html)

    # clean up markdown
    lines = md.strip().split('\n')
    prev = ''
    clean_md = []
    for line in lines:
        if line or prev:
            clean_md.append(line)
            prev = line
    return '\n'.join(clean_md)

class BrowserTool(BaseTool):
    """A web search tool that uses a query or a list of queries provided by the user to perform web search via the SearXNG meta search engine API.
    """
    def __init__(self, 
        base_url: str, 
        embeddings: "BaseEmbeddings",
        tokenizer: Optional["BaseTokenizer"] = None,
        ranker: Optional["BaseRanker"] = None,
        chunk_size: int = 300
    ) -> None:
        """Initializes the BrowserTool with the given parameters.

        Args:
            base_url (str): The base URL of the SearXNG search engine.
            embeddings (BaseEmbeddings): The embeddings model to use for relevant content searching in the search results.
            tokenizer (Optional[BaseTokenizer], optional): The tokenizer to use for token counting for result chunks. If None is provided, the tokenizer from the embedding model will be used. Defaults to None.
            ranker (Optional[BaseRanker], optional): The ranker to use for reranking the search results. If None is given, it will be initialised with ms-marco-TinyBERT. Defaults to None.
            chunk_size (int, optional): The maximum number of tokens in each chunk of the search results. Defaults to 300.
        """
        try:
            from markdownify import markdownify
        except:
            raise ModuleNotFoundError(f'Module "markdownify" not installed. Please install it with `pip`.')
        from ..TextSplitter.sentence_splitter import SentenceTextSplitter
        from ..VectorDatabase.numpy_vectordb import NumpyVectorDatabase
        from ..Reranker.huggingface_ranker import HuggingFaceRanker
        self._base_url = base_url.rstrip('/').removesuffix('search').rstrip('/')
        text_splitter = SentenceTextSplitter(tokenizer=tokenizer if tokenizer else embeddings.tokenizer, chunk_size=chunk_size)
        self._vdb = NumpyVectorDatabase(embeddings=embeddings, text_splitter=text_splitter)
        self._ranker = ranker if ranker else HuggingFaceRanker()
        super().__init__()

    def __call__(self,
        query: Union[str, List[str]],
        top_k: int = 3,
        category: Literal['general', 'news', 'IT'] = 'general',
        language: Literal['all', 'en', 'en-US', 'de', 'it', 'fr', 'es', 'zh', 'hi', 'ja', 'ko', 'ru'] = 'all'
    ) -> List[Dict[str, Any]]:
        """Performs a web search using the provided query or list of queries and returns the top_k results for each provided query.

        Args:
            query (Union[str, List[str]]): The query or list of queries to search for.
            top_k (int, optional): The number of top results to return for each query. Defaults to 3.
            category (Literal['general', 'news', 'IT'], optional): The category of the search results. Defaults to 'general'.
            language (Literal['all', 'en', 'en-US', 'de', 'it', 'fr', 'es', 'zh', 'hi', 'ja', 'ko', 'ru'], optional): The language of the search results. Defaults to 'all'.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the search results.
        """
        from ..Schema.document import Document
        from threading import Thread
        import json
        qs = query if isinstance(query, list) else [query]
        queries = []
        for q in qs:
            if q not in queries:
                queries.append(q)
        request_args = dict(format='json')
        if language != 'all':
            request_args['language'] = language
        if category != 'general':
            request_args['categories'] = [category.lower()]
        
        # Sending requests to API server
        requests = [dict(q=q, **request_args) for q in queries]
        outputs = [None] * len(requests)
        def parse_result(result) -> Dict[str, Any]:
            source = result.get('parsed_url', [])
            source = source[1] if len(source) > 1 else None
            final = dict(
                title=result.get('title', None),
                url=result['url'],
                source=source,
                published_date=result.get('publishedDate', None),
                category=result.get('category', None),
                search_engine=result.get('engine', None),
                content=result.get('content', None)
            )
            return final
        def get_results(i):
            try:
                res = get(url=self._base_url + '/search', params=requests[i])
                res.raise_for_status()
            except:
                res = None
            if res:
                results = json.loads(res.text)['results']
                if isinstance(results, list):
                    results = [parse_result(r) for r in results[:top_k]]
                else:
                    results = []
                outputs[i] = results
            else:
                outputs[i] = []
        
        threads = [Thread(target=get_results, args=[i]) for i in range(len(requests))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        outputs = [[] if o is None else o for o in outputs]

        # Getting contents from the search results
        mds = [[None] * len(i) for i in outputs]
        def get_markdown(i, j):
            mds[i][j] = url_to_markdown(outputs[i][j]['url'])
        threads = [[Thread(target=get_markdown, args=[i, j]) for j in range(len(o))] for i, o in enumerate(outputs)] 
        [[t.start() for t in mt] for mt in threads]
        [[t.join() for t in mt] for mt in threads]
        q_results = [(q, [Document(text=m, metadata=mt) for m, mt in zip(ms, mts) if m.strip() != '']) for q, ms, mts  in zip(queries, mds, outputs)]
        
        self._vdb.clear()
        search_results = []
        for q, res in q_results:
            if len(res) > 0:
                self._vdb.add_documents(res)
                vdb_results = self._vdb.search(q, top_k=top_k + 5)
                ranked_results = self._ranker.rerank(query=q, docs=[vres.doc for vres in vdb_results], top_k=top_k)
                search_results.append(dict(search_query=q, search_results=[rr.doc.model_dump() for rr in ranked_results]))
                self._vdb.clear()
        return search_results
    
    def run(self, **kwargs) -> ToolOutput:
        """Run the tool and structure the output into a ToolOutput object.

        Returns:
            ToolOutput: Structured tool output.
        """
        import json
        outputs = self.__call__(**kwargs)
        urls = []
        for q in outputs:
            for res in q['search_results']:
                if res['metadata']['url'] not in urls:
                    urls.append(res['metadata']['url'])
        return ToolOutput(content=json.dumps(outputs), footnotes=urls)



