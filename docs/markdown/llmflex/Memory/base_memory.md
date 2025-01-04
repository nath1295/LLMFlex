Module llmflex.Memory.base_memory
=================================

Classes
-------

`BaseMemory(title: Optional[str] = None, memory_dir: Optional[str] = None)`
:   Base chat memory class.
    
    Initializes the BaseMemory class.
    
    Args:
        title (Optional[str], optional): The title of the memory. Defaults to None.
        memory_dir (Optional[str], optional): The directory where the memory will be stored. Defaults to None.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.Memory.chat_memory.ChatMemory
    * llmflex.Memory.long_short_term_chat_memory.LongShortTermChatMemory

    ### Static methods

    `from_exist(memory_dir: str) ‑> llmflex.Memory.base_memory.BaseMemory`
    :   Loads a memory instance from an existing directory.
        
        Args:
            memory_dir (str): The directory containing the memory data.
        
        Returns:
            BaseMemory: An instance of the BaseMemory subclass.

    ### Instance variables

    `history: List[Dict[str, Any]]`
    :   Returns the conversation history.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a conversation turn.

    `info: Dict[str, Union[str, Optional[str], float]]`
    :   Returns information about the memory.
        
        Returns:
            Dict[str, Union[str, Optional[str], float]]: A dictionary containing information about the memory.

    `memory_dir: Optional[str]`
    :   Returns the directory where the memory will be stored.
        
        Returns:
            Optional[str]: The directory where the memory will be stored.

    `title: Optional[str]`
    :   Returns the title of the memory.
        
        Returns:
            Optional[str]: The title of the memory.

    ### Methods

    `get_last_n_messages(self, last_n: int = 3) ‑> List[Dict[str, Any]]`
    :   Returns the last n messages from the conversation history.
        
        Args:
            last_n (int, optional): The number of messages to retrieve. Defaults to 3.
        
        Raises:
            ValueError: If last_n is less than 1.
        
        Returns:
            List[Dict[str, Any]]: A list of the last n messages.

    `get_messages_by_token_limit(self, tokenizer: "'BaseTokenizer'", token_limit: int) ‑> List[Dict[str, Any]]`
    :   Returns messages from the conversation history that do not exceed the given token limit.
        
        Args:
            tokenizer (BaseTokenizer): The tokenizer to use for token counting.
            token_limit (int): The maximum number of tokens allowed in the returned messages.
        
        Returns:
            List[Dict[str, Any]]: A list of messages that do not exceed the token limit.

    `remove_last_message(self) ‑> None`
    :   Remove the latest message.

    `save_messages(self, messages: Union[Dict[str, Any], List[Dict[str, Any]]]) ‑> None`
    :   Saves the given messages to the memory.
        
        Args:
            messages (Union[Dict[str, Any], List[Dict[str, Any]]]): A dictionary or list of dictionaries representing the messages to be saved.

    `set_title(self, title: Optional[str]) ‑> None`
    :   Sets the title of the memory.
        
        Args:
            title (Optional[str]): The new title for the memory.