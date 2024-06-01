Module llmflex.Memory.long_short_memory
=======================================

Classes
-------

`LongShortTermChatMemory(chat_id: str, embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit], llm: Optional[llmflex.Models.Cores.base_core.BaseLLM], ranker: Optional[llmflex.Rankers.base_ranker.BaseRanker] = None, text_splitter: Optional[llmflex.TextSplitters.base_text_splitter.BaseTextSplitter] = None, ts_lang_model: str = 'en_core_web_sm', chunk_size: int = 400, chunk_overlap: int = 40, from_exist: bool = True, system: Optional[str] = None)`
:   Base class for chat memory.
        
    
    Initialising the memory class.
    
    Args:
        chat_id (str): Chat ID.
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit for the vector database for storing chat history.
        llm (Optional[BaseLLM]): LLM for counting tokens.
        ranker (Optional[BaseRanker], optional): Reranker for long term memory retrieval. Defaults to None.
        text_splitter (Optional[BaseTextSplitter], optional): Text splitter to use. If None given, it will be created with other arguments. Defaults to None.
        ts_lang_model (str, optional): Language model for the sentence text splitter. Defaults to 'en_core_web_sm'.
        chunk_size (int, optional): Chunk size for the text splitter. Defaults to 400.
        chunk_overlap (int, optional): Chunk overlap for the text splitter. Defaults to 40.
        from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.
        system (Optional[str], optional): System message for the chat. If None is given, the default system message or the stored system message will be used. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Memory.base_memory.BaseChatMemory

    ### Instance variables

    `embeddings: llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit`
    :   Embeddings toolkit.
        
        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit.

    `ranker: llmflex.Rankers.base_ranker.BaseRanker`
    :   Reranker.
        
        Returns:
            BaseRanker: Reranker.

    `text_splitter: llmflex.TextSplitters.sentence_token_text_splitter.SentenceTokenTextSplitter`
    :   Sentence text splitter.
        
        Returns:
            SentenceTokenTextSplitter: Sentence text splitter.

    `vectordb: llmflex.VectorDBs.faiss_vectordb.FaissVectorDatabase`
    :   Vector database for saving the chat history.
        
        Returns:
            FaissVectorDatabase: Vector database for saving the chat history.

    ### Methods

    `get_long_term_memory(self, query: str, llm: Type[llmflex.Models.Cores.base_core.BaseLLM], recent_history: Union[List[str], List[Tuple[str, str]], ForwardRef(None)] = None, token_limit: int = 400, similarity_score_threshold: float = 0.2, relevance_score_threshold: float = 0.8) ‑> List[Dict[str, Any]]`
    :   Retriving the long term memory with the given query. Usually used together with get_token_memory.
        
        Args:
            query (str): Search query for the vector database. Usually the latest user input.
            llm (Type[BaseLLM]): LLM to count tokens.
            recent_history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): List of interactions in the short term memory to skip in the long term memory. Defaults to None.
            token_limit (int, optional): Maximum number of tokens in the long term memory. Defaults to 400.
            similarity_score_threshold (float, optional): Minimum threshold for similarity score, shoulbe be between 0 to 1. Defaults to 0.2.
            relevance_score_threshold (float, optional): Minimum threshold for relevance score for the reranker, shoulbe be between 0 to 1. Defaults to 0.8.
        
        Returns:
            List[Dict[str, Any]]: List of chunks related to the query and their respective speaker.