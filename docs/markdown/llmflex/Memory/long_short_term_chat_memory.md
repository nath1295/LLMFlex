Module llmflex.Memory.long_short_term_chat_memory
=================================================

Classes
-------

`LongShortTermChatMemory(embeddings: "'BaseEmbeddings'", ranker: "Optional['BaseRanker']" = None, text_splitter: "Optional['BaseTextSplitter']" = None, vdb_type: VDB_TYPE = 'numpy', title: Optional[str] = None, memory_dir: Optional[str] = None, **kwargs)`
:   Long short term chat memory class.
    
    Initializes the LongShortTermChatMemory class.
    
    Args:
        embeddings (BaseEmbeddings): The embeddings model to use for encoding and decoding.
        ranker (Optional[BaseRanker], optional): The ranker model to use for ranking the messages during search. Defaults to None.
        text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting the longer messages. Defaults to None.
        vdb_type (VDB_TYPE, optional): The type of vector database to use. Defaults to 'numpy'.
        title (Optional[str], optional): The title of the memory. Defaults to None.
        memory_dir (Optional[str], optional): The directory to store the memory. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Memory.base_memory.BaseMemory
    * abc.ABC

    ### Static methods

    `from_exist(embeddings: "'BaseEmbeddings'", memory_dir: str, ranker: "Optional['BaseRanker']" = None, text_splitter: "Optional['BaseTextSplitter']" = None)`
    :   Initializes the LongShortTermChatMemory class from an existing memory directory.
        
        Args:
            embeddings (BaseEmbeddings): The embeddings model to use for encoding and decoding.
            memory_dir (str): The directory that stores the memory.
            ranker (Optional[BaseRanker], optional): The ranker model to use for ranking the messages during search. Defaults to None.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting the longer messages. Defaults to None.
        
        Returns:
            LongShortTermChatMemory: An instance of LongShortTermChatMemory initialized from the existing vector database.

    ### Methods

    `get_related_messages(self, query: str, token_limit: int = 300, score_threshold: float = 0.0, exclude_messages: Optional[List[Dict[str, Any]]] = None) ‑> List[Dict[str, str]]`
    :   Retrieves related messages based on a query.
        
        Args:
            query (str): The query to search for related messages.
            token_limit (int, optional): The maximum number of tokens to consider for each message. Defaults to 300.
            score_threshold (float, optional): The minimum score required for a message to be considered related. Defaults to 0.0.
            exclude_messages (Optional[List[Dict[str, Any]]], optional): A list of messages to exclude from the search. Defaults to None.
        
        Returns:
            List[Dict[str, str]]: A list of related messages, each represented as a dictionary with 'role', 'content', and 'tool_calls' keys.