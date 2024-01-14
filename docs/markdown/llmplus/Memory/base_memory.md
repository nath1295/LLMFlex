Module llmplus.Memory.base_memory
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

`BaseChatMemory(title: str, from_exist: bool = True)`
:   Base class for chat memory.
        
    
    Initialising the memory class.
    
    Args:
        title (str): Title of the chat.
        from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.

    ### Descendants

    * llmplus.Memory.long_short_memory.LongShortTermChatMemory

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

    `title: str`
    :   Chat title.
        
        Returns:
            str: Chat title.

    ### Methods

    `clear(self) ‑> None`
    :   Empty the whole chat history.

    `get_recent_memory(self, k: int = 3) ‑> List[Tuple[str, str]]`
    :   Get the last k interactions as a list.
        
        Args:
            k (int, optional): Maximum number of latest interactions. Defaults to 3.
        
        Returns:
            List[Tuple[str, str]]: List of interactions.

    `get_token_memory(self, llm: Type[llmplus.Models.Cores.base_core.BaseLLM], token_limit: int = 400) ‑> List[str]`
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