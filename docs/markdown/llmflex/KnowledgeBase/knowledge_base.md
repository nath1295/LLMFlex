Module llmflex.KnowledgeBase.knowledge_base
===========================================

Functions
---------

    
`get_new_kb_id() ‑> str`
:   Get a new id for a new knowledge base.
    
    Returns:
        str: The new kb_id.

    
`knowledge_base_dir() ‑> str`
:   Directory to store knowlege base.
    
    Returns:
        str: Directory to store knowlege base.

    
`list_knowledge_base() ‑> List[str]`
:   List the existing knowledge base.
    
    Returns:
        List[str]: List of the existing knowledge bases.

    
`load_docx(file_dir: str) ‑> List[llmflex.Schemas.documents.Document]`
:   Load a docx file as list of documents for the knowledge base to add.
    
    Args:
        file_dir (str): Full directory of the docx file.
    
    Returns:
        List[Document]: List of documents from the docx file.

    
`load_file(file_dir: str, filetype: Literal['auto', 'markdown', 'docx', 'pdf'] = 'auto') ‑> List[llmflex.Schemas.documents.Document]`
:   Load a text-based file as list of docments.
    
    Args:
        file_dir (str): Full directory of the file.
        filetype (Literal[&#39;auto&#39;, &#39;markdown&#39;, &#39;docx&#39;, &#39;pdf&#39;], optional): The type of file to be loaded. If auto is set, it will be determined by the suffix of the file. Defaults to 'auto'.
    
    Returns:
        List[Document]: List of documents from the pdf file.

    
`load_markdown(file_dir: str) ‑> List[llmflex.Schemas.documents.Document]`
:   Load a markdown file as list of documents for the knowledge base to add.
    
    Args:
        file_dir (str): Full directory of the markdown file.
    
    Returns:
        List[Document]: List of documents from the markdown file.

    
`load_pdf(file_dir: str) ‑> List[llmflex.Schemas.documents.Document]`
:   Load a pdf file as list of documents for the knowledge base to add.
    
    Args:
        file_dir (str): Full directory of the pdf file.
    
    Returns:
        List[Document]: List of documents from the pdf file.

Classes
-------

`KnowledgeBase(kb_id: str, embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit], llm: Optional[llmflex.Models.Cores.base_core.BaseLLM], ranker: Optional[llmflex.Rankers.base_ranker.BaseRanker] = None, text_splitter: Optional[llmflex.TextSplitters.base_text_splitter.BaseTextSplitter] = None, ts_lang_model: str = 'en_core_web_sm', chunk_size: int = 400, chunk_overlap: int = 40)`
:   Class to store any text as knowledge for querying.
        
    
    Initialise the knowlege base.
    
    Args:
        kb_id (str): A unique identifier for the knowledge base starting with "kb_".
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit for the vector database.
        llm (Optional[BaseLLM]): LLM for counting tokens. If not given, the embedding model tokenizer will be used to count tokens.
        ranker (Optional[BaseRanker], optional): Reranker to rerank sementic search results. Defaults to None.
        text_splitter (Optional[BaseTextSplitter], optional): Text splitter to split documents. If None is given, it will be created with the token counting function. Defaults to None.
        ts_lang_model (str, optional): Text splitter language model to use if text_splitter is not provided. Defaults to 'en_core_web_sm'.
        chunk_size (int, optional): Chunk size of the text splitter if text_splitter is not provided. Defaults to 400.
        chunk_overlap (int, optional): Chunk overlap of the text splitter if text_splitter is not provided. Defaults to 40.

    ### Instance variables

    `count_fn: Callable[[str], int]`
    :   Function to count number of tokens in a string.
        
        Returns:
            Callable[[str], int]: Function to count number of tokens in a string.

    `embeddings: llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit`
    :   Embeddings toolkit for the vector database.
        
        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit for the vector database.

    `files: List[Tuple[str, str]]`
    :   List of files and their respective directories.
        Returns:
            List[Tuple[str, str]]: List of files and their respective directories.

    `kb_id: str`
    :   Knowledge base id.
        
        Returns:
            str: Knowledge base id.

    `knowledge_base_dir: str`
    :   Directory for the knowledge base.
        
        Returns:
            str: Directory for the knowledge base.

    `ranker: llmflex.Rankers.base_ranker.BaseRanker`
    :   Reranker for search results.
        
        Returns:
            BaseRanker: Reranker for search results.

    `text_splitter: llmflex.TextSplitters.base_text_splitter.BaseTextSplitter`
    :   Text splitter for the knowledge base.
        
        Returns:
            BaseTextSplitter: Text splitter for the knowledge base.

    `vector_db: llmflex.VectorDBs.base_vectordb.BaseVectorDatabase`
    :   Vector database for the knowledge base.
        
        Returns:
            BaseVectorDatabase: Vector database for the knowledge base.

    ### Methods

    `add_documents(self, docs: List[llmflex.Schemas.documents.Document], mode: Literal['update', 'append'] = 'update') ‑> None`
    :   Adding documents into the knowledge base. In the metadata of the file, it should contain at least filename and file_dir.
        
        Args:
            docs (List[Document]): List of documents to add.
            mode (Literal['update', 'append'], optional): Way of adding documents. Either updating/add the files or append on existing files. Defaults to 'update'.

    `clear(self) ‑> None`
    :   Clear the entire knowledge base. Use it with caution.

    `search(self, query: str, top_k=3, token_limit: Optional[int] = None, fetch_k: int = 30, count_fn: Optional[Callable[[str], int]] = None, relevance_score_threshold: float = 0.8) ‑> List[llmflex.Schemas.documents.RankResult]`
    :   Searching for related information from the knowledge base.
        
        Args:
            query (str): Search query.
            top_k (int, optional): Maximum number of result. If token_limit is not None, token_limit will be used instead. Defaults to 3.
            token_limit (Optional[int], optional): Maximum number of tokens for the search results. Defaults to None.
            fetch_k (int, optional): Number of results to fetch from the vector database before reranking. Defaults to 30.
            count_fn (Optional[Callable[[str], int]], optional): Function to count the number of tokens if token_limit is not None. If None is given, the count_fn from the knowledge base class will be used. Defaults to None.
            relevance_score_threshold (float, optional): Minumum score for the reranking. Defaults to 0.8.
        
        Returns:
            List[RankResult]: List of search results.