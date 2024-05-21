Module llmflex.Memory.base_memory
=================================

Functions
---------

    
`chat_memory_home() ‑> str`
:   Return the default directory for saving chat memories.
    
    Returns:
        str: The default directory for saving chat memories.

    
`list_chat_dirs() ‑> List[str]`
:   List the directories of all chat memories.
    
    Returns:
        List[str]: List of directories of all chat memories.

    
`list_titles() ‑> List[str]`
:   Return a list of chat titles.
    
    Returns:
        List[str]: List of chat titles, sorted by last update descendingly.

    
`title_dir_map() ‑> Dict[str, str]`
:   Return a dictionary with chat titles as keys and their respective directories as values.
    
    Returns:
        Dict[str, str]: A dictionary with chat titles as keys and their respective directories as values.

Classes
-------

`BaseChatMemory(title: str, from_exist: bool = True, system: Optional[str] = None)`
:   Base class for chat memory.
        
    
    Initialising the memory class.
    
    Args:
        title (str): Title of the chat.
        from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.
        system (Optional[str], optional): System message for the chat. If None is given, the default system message or the stored system message will be used. Defaults to None.

    ### Descendants

    * llmflex.Memory.assistant_long_term_memory.AssistantLongTermChatMemory
    * llmflex.Memory.long_short_memory.LongShortTermChatMemory
    * llmflex.Memory.longshort_memory.LongShortTermChatMemory

    ### Instance variables

    `chat_dir: str`
    :   Directory of the chat.
        
        Returns:
            str: Directory of the chat.

    `history: List[Tuple[str, str]]`
    :   Entire chat history.
        
        Returns:
            List[Tuple[str, str]]: Entire chat history.

    `info: Dict[str, Any]`
    :   Information of the chat.
        
        Returns:
            Dict[str, Any]: Information of the chat.

    `interaction_count: int`
    :   Number of interactions.
        
        Returns:
            int: Number of interactions.

    `system: str`
    :   Default system message of the memory.
        
        Returns:
            str: Default system message of the memory.

    `title: str`
    :   Chat title.
        
        Returns:
            str: Chat title.

    ### Methods

    `clear(self) ‑> None`
    :   Empty the whole chat history.

    `create_prompt_with_memory(self, user: str, prompt_template: llmflex.Prompts.prompt_template.PromptTemplate, llm: Type[llmflex.Models.Cores.base_core.BaseLLM], system: Optional[str] = None, recent_token_limit: int = 200, knowledge_base: Optional[llmflex.KnowledgeBase.knowledge_base.KnowledgeBase] = None, relevance_token_limit: int = 200, relevance_score_threshold: float = 0.8, **kwargs) ‑> str`
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
        
        
        Returns:
            str: The full chat prompt.

    `get_recent_memory(self, k: int = 3) ‑> List[Tuple[str, str]]`
    :   Get the last k interactions as a list.
        
        Args:
            k (int, optional): Maximum number of latest interactions. Defaults to 3.
        
        Returns:
            List[Tuple[str, str]]: List of interactions.

    `get_token_memory(self, llm: Type[llmflex.Models.Cores.base_core.BaseLLM], token_limit: int = 400) ‑> List[str]`
    :   Get the latest conversation limited by number of tokens.
        
        Args:
            llm (Type[BaseLLM]): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens allowed. Defaults to 400.
        
        Returns:
            List[str]: List of most recent messages.

    `remove_last_interaction(self) ‑> None`
    :   Remove the latest interaction.

    `save(self) ‑> None`
    :   Save the current state of the memory.

    `save_interaction(self, user_input: str, assistant_output: str, **kwargs) ‑> None`
    :   Saving an interaction to the memory.
        
        Args:
            user_input (str): User input.
            assistant_output (str): Chatbot output.

    `update_system_message(self, system: str) ‑> None`
    :   Update the default system message for the memory.
        
        Args:
            system (str): New system message.