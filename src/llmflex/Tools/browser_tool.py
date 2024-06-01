from .tool_utils import BaseTool
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..Models.Cores.base_core import BaseLLM
from ..Rankers.base_ranker import BaseRanker
from typing import Optional, Dict, Any, List

class BrowserTool(BaseTool):
    """Tool for browsing contents via the DuckDuckGo search engine given any search query. The output will be the most relevant chunks of content found from the search engine according to the search query.
    """
    def __init__(self, embeddings: BaseEmbeddingsToolkit, llm: Optional[BaseLLM] = None, ranker: Optional[BaseRanker] = None) -> None:
        """Initialising the tool.

        Args:
            embeddings (BaseEmbeddingsToolkit): Embeddings toolkit for the vector database.
            llm (Optional[BaseLLM], optional): LLM to count number of tokens for each chunk. Defaults to None.
            ranker (Optional[BaseRanker], optional): Reranker to rerank results. If none is given, results will not be reranked after the search on the vector database. Defaults to None.
        """
        from ..VectorDBs.faiss_vectordb import FaissVectorDatabase
        self.embeddings = embeddings
        self.vdb = FaissVectorDatabase.from_documents(embeddings=self.embeddings, docs=[])
        self.llm = llm
        self.ranker = ranker
        super().__init__()

    def __call__(self, search_query: str, max_num_results: int = 3) -> Dict[str, List[str]]:
        """Entry point of the tool.

        Args:
            search_query (str): Search query to browse on DuckDuckGo.
            max_num_results (int, optional): Maximum number of relevant chunks of contents from the search results. Defaults to 3.

        Returns:
            Dict[str, List[str]]: The most relevant chunks on contents along with their resepctive URLs.
        """
        import gc
        from .web_search_utils import ddg_search, get_markdown, create_content_chunks
        from ..Schemas.documents import Document
        results = ddg_search(query=search_query, urls_only=False)
        contents = list(map(lambda x: get_markdown(x['href'], as_list=True), results))
        count_fn = self.llm.get_num_tokens if self.llm is not None else self.embeddings.tokenizer.get_num_tokens
        docs = []
        for i, c in enumerate(contents):
            if c is not None:
                doc = create_content_chunks(contents=c, token_count_fn=count_fn)
                doc = list(map(lambda x: Document(index=x, metadata=results[i]), doc))
                docs.extend(doc)
        self.vdb.add_documents(docs, split_text=False)
        if self.ranker:
            chunks = self.vdb.search(query=search_query, top_k=max(int(max_num_results * 2), 15), index_only=False)
            self.vdb.clear()
            res = self.ranker.rerank(query=search_query, elements=chunks, top_k=max_num_results)
            res = list(map(lambda x: x.to_dict(), res))
            chunks = list(map(lambda x: x['index'], res))
            sources = list(set(map(lambda x: x['metadata']['href'], res)))
        else:
            chunks = self.vdb.search(query=search_query, top_k=max_num_results, index_only=False)
            self.vdb.clear()
            res = chunks.copy()
            chunks = list(map(lambda x: x['index'], res))
            sources = list(set(map(lambda x: x['metadata']['href'], res)))
        gc.collect()
        output = dict(relevant_contents=chunks, footnote=sources)
        return output
    

    