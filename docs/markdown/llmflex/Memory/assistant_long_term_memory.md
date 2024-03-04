Module llmflex.Memory.assistant_long_term_memory
================================================

Functions
---------

    
`create_long_assistant_memory_prompt(user: str, prompt_template: llmflex.Prompts.prompt_template.PromptTemplate, llm: Type[llmflex.Models.Cores.base_core.BaseLLM], memory: llmflex.Memory.assistant_long_term_memory.AssistantLongTermChatMemory, system: str = 'This is a conversation between a human user and a helpful AI assistant.', short_token_limit: int = 200, long_token_limit: int = 200, score_threshold: float = 0.5) ‑> str`
:   Wrapper function to create full chat prompts using the prompt template given, with long term memory included in the prompt. 
    
    Args:
        user (str): User newest message.
        prompt_template (PromptTemplate): Prompt template to use.
        llm (Type[BaseLLM]): LLM for counting tokens.
        memory (AssistantLongTermChatMemory): The memory class with long short term functionalities.
        system (str, optional): System message for the conversation. Defaults to DEFAULT_SYSTEM_MESSAGE.
        short_token_limit (int, optional): Maximum number of tokens for short term memory. Defaults to 200.
        long_token_limit (int, optional): Maximum number of tokens for long term memory. Defaults to 200.
        score_threshold (float, optional): Minimum relevance score to be included in long term memory. Defaults to 0.5.
    
    Returns:
        str: The full chat prompt.

Classes
-------

`AssistantLongTermChatMemory(title: str, embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit], text_splitter: llmflex.TextSplitters.sentence_token_text_splitter.SentenceTokenTextSplitter, from_exist: bool = True)`
:   Base class for chat memory.
        
    
    Initialising the memory class.
    
    Args:
        title (str): Title of the chat.
        from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.

    ### Ancestors (in MRO)

    * llmflex.Memory.base_memory.BaseChatMemory

    ### Instance variables

    `embeddings: llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit`
    :   Embeddings toolkit.
        
        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit.

    `text_splitter: llmflex.TextSplitters.sentence_token_text_splitter.SentenceTokenTextSplitter`
    :   Sentence text splitter.
        
        Returns:
            SentenceTokenTextSplitter: Sentence text splitter.

    `vectordb: llmflex.VectorDBs.faiss_vectordb.FaissVectorDatabase`
    :   Vector database for saving the chat history.
        
        Returns:
            FaissVectorDatabase: Vector database for saving the chat history.

    ### Methods

    `get_long_term_assistant_memory(self, query: str, recent_history: Union[List[str], List[Tuple[str, str]]], llm: Type[llmflex.Models.Cores.base_core.BaseLLM], token_limit: int = 400, score_threshold: float = 0.2) ‑> List[str]`
    :   Retriving the long term memory with the given query. Usually used together with get_token_memory.
        
        Args:
            query (str): Search query for the vector database. Usually the latest user input.
            recent_history (Union[List[str], List[Tuple[str, str]]]): List of interactions in the short term memory to skip in the long term memory.
            llm (Type[BaseLLM]): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens in the long term memory. Defaults to 400.
            score_threshold (float, optional): Minimum threshold for similarity score, shoulbe be between 0 to 1. Defaults to 0.2.
        
        Returns:
            List[str]: List of assistant chunks related to the query.