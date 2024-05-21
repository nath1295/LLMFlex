Module llmflex.Memory.longshort_memory
======================================

Classes
-------

`LongShortTermChatMemory(title: str, embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit], llm: Optional[llmflex.Models.Cores.base_core.BaseLLM], ranker: Optional[llmflex.Rankers.base_ranker.BaseRanker] = None, text_splitter: Optional[llmflex.TextSplitters.base_text_splitter.BaseTextSplitter] = None, ts_lang_model: str = 'en_core_web_sm', chunk_size: int = 400, chunk_overlap: int = 40, from_exist: bool = True, system: Optional[str] = None)`
:   Base class for chat memory.
        
    
    Initialising the memory class.
    
    Args:
        title (str): Title of the chat.
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit for the vector database for storing chat history.
        llm (Optional[BaseLLM]): LLM for counting tokens.
        ranker (Optional[BaseRanker], optional): Reranker for long term memory retrieval. Defaults to None.
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

    `create_prompt_with_memory(self, user: str, prompt_template: llmflex.Prompts.prompt_template.PromptTemplate, llm: Type[llmflex.Models.Cores.base_core.BaseLLM], system: Optional[str] = None, recent_token_limit: int = 200, knowledge_base: Optional[llmflex.KnowledgeBase.knowledge_base.KnowledgeBase] = None, relevance_token_limit: int = 200, relevance_score_threshold: float = 0.8, similarity_score_threshold: float = 0.5, **kwargs) ‑> str`
    :   Wrapper function to create full chat prompts using the prompt template given, with long term memory included in the prompt. 
        
        Args:
            user (str): User newest message.
            prompt_template (PromptTemplate): Prompt template to use.
            llm (Type[BaseLLM]): LLM for counting tokens.
            system (Optional[str], optional): System message to override the default system message for the memory. Defaults to None.
            recent_token_limit (int, optional): Maximum number of tokens for recent term memory. Defaults to 200.
            knowledge_base (Optional[KnowledgeBase]): Knowledge base that helps the assistant to answer questions. Defaults to None.
            relevance_token_limit (int, optional): Maximum number of tokens for search results from the knowledge base if a knowledge base is given. Defaults to 200.
            relevance_score_threshold (float, optional): Reranking score threshold for knowledge base search if a knowledge base is given. Defaults to 0.8.
            similarity_score_threshold (float, optional): Long term memory similarity score threshold. Defaults to 0.5.
        
        
        Returns:
            str: The full chat prompt.

    `get_long_term_memory(self, query: str, recent_history: Union[List[str], List[Tuple[str, str]]], llm: Type[llmflex.Models.Cores.base_core.BaseLLM], token_limit: int = 400, similarity_score_threshold: float = 0.2, relevance_score_threshold: float = 0.8) ‑> List[Dict[str, Any]]`
    :   Retriving the long term memory with the given query. Usually used together with get_token_memory.
        
        Args:
            query (str): Search query for the vector database. Usually the latest user input.
            recent_history (Union[List[str], List[Tuple[str, str]]]): List of interactions in the short term memory to skip in the long term memory.
            llm (Type[BaseLLM]): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens in the long term memory. Defaults to 400.
            similarity_score_threshold (float, optional): Minimum threshold for similarity score, shoulbe be between 0 to 1. Defaults to 0.2.
            relevance_score_threshold (float, optional): Minimum threshold for relevance score for the reranker, shoulbe be between 0 to 1. Defaults to 0.8.
        
        Returns:
            List[Dict[str, Any]]: List of chunks related to the query and their respective speaker.