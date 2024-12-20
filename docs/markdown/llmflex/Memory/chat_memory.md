Module llmflex.Memory.chat_memory
=================================

Classes
-------

`ChatMemory(title: Optional[str] = None, memory_dir: Optional[str] = None)`
:   Basic chat memory class.
    
    Initializes the ChateMemory class.
    
    Args:
        title (Optional[str], optional): The title of the memory. Defaults to None.
        memory_dir (Optional[str], optional): The directory where the memory will be stored. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Memory.base_memory.BaseMemory
    * abc.ABC

    ### Static methods

    `from_exist(memory_dir: str) ‑> llmflex.Memory.chat_memory.ChatMemory`
    :   Loads a memory instance from an existing directory.
        
        Args:
            memory_dir (str): The directory containing the memory data.
        
        Returns:
            ChatMemory: An instance of the ChatMemory class.