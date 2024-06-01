Module llmflex.Memory.base_memory
=================================

Functions
---------

    
`chat_memory_home() ‑> str`
:   Return the default directory for saving chat memories.
    
    Returns:
        str: The default directory for saving chat memories.

    
`get_dir_from_id(chat_id: str) ‑> str`
:   Geet the memory directory given the chat ID.
    
    Args:
        chat_id (str): Chat ID.
    
    Returns:
        str: Memory directory.

    
`get_new_chat_id() ‑> str`
:   Get an unused chat id.
    
    Returns:
        str: New chat id.

    
`get_title_from_id(chat_id: str) ‑> str`
:   Getting the title from Chat ID.
    
    Args:
        chat_id (str): Chat ID.
    
    Returns:
        str: Title of the memory.

    
`list_chat_dirs() ‑> List[str]`
:   List the directories of all chat memories.
    
    Returns:
        List[str]: List of directories of all chat memories.

    
`list_chat_ids() ‑> List[str]`
:   Return a list of existing chat ids.
    
    Returns:
        List[str]: Return a list of existing chat ids, sorted by last update descendingly.

    
`list_titles() ‑> List[str]`
:   Return a list of chat titles.
    
    Returns:
        List[str]: List of chat titles, sorted by last update descendingly.

Classes
-------

`BaseChatMemory(chat_id: str, from_exist: bool = True, system: Optional[str] = None)`
:   Base class for chat memory.
        
    
    Initialising the memory class.
    
    Args:
        chat_id (str): Chat ID.
        from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.
        system (Optional[str], optional): System message for the chat. If None is given, the default system message or the stored system message will be used. Defaults to None.

    ### Descendants

    * llmflex.Memory.long_short_memory.LongShortTermChatMemory

    ### Instance variables

    `chat_dir: str`
    :   Directory of the chat.
        
        Returns:
            str: Directory of the chat.

    `chat_id: str`
    :   Unique identifier of the chat memory.
        
        Returns:
            str: Unique identifier of the chat memory.

    `history: List[Tuple[str, str]]`
    :   Entire chat history.
        
        Returns:
            List[Tuple[str, str]]: Entire chat history.

    `history_dict: List[Dict[str, Any]]`
    :   Entire history as dictionaries.
        
        Returns:
            List[Dict[str, Any]]: Entire history as dictionaries.

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

    `update_title(self, title: str) ‑> None`
    :   Update the title of the memory.
        
        Args:
            title (str): New chat memory title.