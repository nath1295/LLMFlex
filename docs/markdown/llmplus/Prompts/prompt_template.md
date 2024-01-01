Module llmplus.Prompts.prompt_template
======================================

Classes
-------

`PromptTemplate(system_prefix: str, system_suffix: str, human_prefix: str, human_suffix: str, ai_prefix: str, ai_suffix: str, wrapper: List[str], stop: Optional[List[str]] = None)`
:   Class for storing prompt format presets.
        
    
    Initialising the chat prompt class.
    
    Args:
        system_prefix (str): System message prefix.
        system_suffix (str): System message suffix.
        human_prefix (str): User message prefix.
        human_suffix (str): User message suffix.
        ai_prefix (str): Chatbot message prefix.
        ai_suffix (str): Chatbot message suffix.
        wrapper (List[str]): Wrapper for start and end of conversation history.
        stop (Optional[List[str]], optional): List of stop strings for the llm. If None is given, the human_prefix will be used. Defaults to None.

    ### Static methods

    `from_dict(format_dict: Dict[str, Any], template_name: Optional[str] = None) ‑> llmplus.Prompts.prompt_template.PromptTemplate`
    :   Initialise the prompt template from a dictionary.
        
        Args:
            format_dict (Dict[str, Any]): Dictionary of the prompt format.
            template_name (Optional[str], optional): Name of the template. Defaults to None.
        
        Returns:
            PromptTemplate: The initialised PromptTemplate instance.

    `from_json(file_dir: str) ‑> llmplus.Prompts.prompt_template.PromptTemplate`
    :   Initialise the prompt template from a json file.
        
        Args:
            file_dir (str): json file path of the prompt format.
        
        Returns:
            PromptTemplatet: The initialised PromptTemplate instance.

    `from_preset(style: "Literal['Default Chat', 'Default Instruct', 'Llama 2 Chat', 'Vicuna 1.1 Chat', 'ChatML Chat', 'Zephyr Chat']") ‑> llmplus.Prompts.prompt_template.PromptTemplate`
    :   Initialise the prompt template from a preset.
        
        Args:
            style (Literal[&#39;Default Chat&#39;, &#39;Default Instruct&#39;, &#39;Llama 2 Chat&#39;, &#39;Vicuna 1.1 Chat&#39;, &#39;ChatML Chat&#39;, &#39;Zephyr Chat&#39;]): Format of the prompt.
        
        Returns:
            PromptTemplate: The initialised PromptTemplate instance.

    ### Instance variables

    `ai_prefix: str`
    :

    `ai_suffix: str`
    :

    `human_prefix: str`
    :

    `human_suffix: str`
    :

    `stop: List[str]`
    :

    `system_prefix: str`
    :

    `system_suffix: str`
    :

    `template_name: str`
    :   Name of the template.
        
        Returns:
            str: Name of the template.

    `wrapper: List[str]`
    :

    ### Methods

    `create_chat_prompt(self, user: str, system: str = 'This is a conversation between a human user and a helpful AI assistant.', history: List[List[str]] = []) ‑> str`
    :   Creating the full chat prompt.
        
        Args:
            user (str): Latest user input.
            system (str, optional): System message. Defaults to DEFAULT_SYSTEM_MESSAGE.
            history (List[List[str]], optional): List of conversation history. Defaults to [].
        
        Returns:
            str: The full prompt.

    `format_history(self, history: List[List[str]], use_wrapper: bool = True) ‑> str`
    :   Formatting a list of conversation history into a full string of conversation history.
        
        Args:
            history (List[List[str]]): List of conversation history. 
            use_wrapper (bool, optional): Whether to format the conversation history with the wrappers. Defaults to True.
        
        Returns:
            str: Full string of conversation history.

    `to_dict(self, return_raw_stop: bool = True) ‑> Dict[str, Any]`
    :   Export the class as a dictionary.
        
        Args:
            return_raw_stop (bool, optional): Whether to return the stop list or the raw input stop value of the PromptTemplate instance.
        
        Returns:
            Dict[str, Any]: Prompt format as a dictionary.