Module llmflex.Prompts.prompt_template
======================================

Classes
-------

`PromptTemplate(template: str, eos_token: Optional[str], bos_token: Optional[str], stop: Optional[List[str]] = None, force_real_template: bool = False)`
:   Class for storing prompt format presets.
        
    
    Initialising the chat prompt class.
    
    Args:
        template (str): Jinja2 template.
        eos_token (Optional[str]): EOS token string.
        bos_token (Optional[str]): BOS token string.
        stop (Optional[List[str]], optional): List of stop strings for the llm. If None is given, the EOS token string will be used. Defaults to None.
        force_real_template (bool, optional): Whether to render the given template. For most templates it has no effects. Only for some restrictive templates like llama2. Defaults to False.

    ### Static methods

    `from_dict(format_dict: Dict[str, Any], template_name: Optional[str] = None) ‑> llmflex.Prompts.prompt_template.PromptTemplate`
    :   Initialise the prompt template from a dictionary.
        
        Args:
            format_dict (Dict[str, Any]): Dictionary of the prompt format.
            template_name (Optional[str], optional): Name of the template. Defaults to None.
        
        Returns:
            PromptTemplate: The initialised PromptTemplate instance.

    `from_json(file_dir: str) ‑> llmflex.Prompts.prompt_template.PromptTemplate`
    :   Initialise the prompt template from a json file.
        
        Args:
            file_dir (str): json file path of the prompt format.
        
        Returns:
            PromptTemplatet: The initialised PromptTemplate instance.

    `from_preset(style: PRESET_FORMATS, force_real_template: bool = False) ‑> llmflex.Prompts.prompt_template.PromptTemplate`
    :   Initialise the prompt template from a preset.
        
        Args:
            style (PRESET_FORMATS): Format of the prompt.
            force_real_template (bool, optional): Whether to render the given template. For most templates it has no effects. Only for some restrictive templates like llama2. Defaults to False.
        
        Returns:
            PromptTemplate: The initialised PromptTemplate instance.

    ### Instance variables

    `bos_token: Optional[str]`
    :

    `eos_token: Optional[str]`
    :

    `rendered_template: jinja2.environment.Environment`
    :

    `stop: List[str]`
    :

    `template: str`
    :

    `template_name: str`
    :   Name of the template.
        
        Returns:
            str: Name of the template.

    ### Methods

    `create_custom_prompt(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) ‑> str`
    :   Creating a custom prompt with your given list of messages. Each message should contain a dictionary with the key "role" and "content".
        
        Args:
            messages (List[Dict[str, str]]): List of messages. Each message should contain a dictionary with the key "role" and "content".
            add_generation_prompt (bool, optional): Whether to add the assistant tokens at the end of the prompt. Defaults to True.
        
        Returns:
            str: The full prompt given your messages.

    `create_prompt(self, user: str, system: str = 'This is a conversation between a human user and a helpful AI assistant.', history: Optional[Union[List[str], List[Tuple[str, str]]]] = None) ‑> str`
    :   Creating the full chat prompt.
        
        Args:
            user (str): Latest user input.
            system (str, optional): System message. Defaults to DEFAULT_SYSTEM_MESSAGE.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): List of conversation history. Defaults to None.
        
        Returns:
            str: The full prompt.

    `format_history(self, history: Union[List[str], List[Tuple[str, str]]], return_list: bool = False) ‑> Union[str, List[Dict[str, str]]]`
    :   Formatting a list of conversation history into a full string of conversation history or a list of messages for the Jinja template to render.
        
        Args:
            history (Union[List[str], List[Tuple[str, str]]]): List of conversation history. 
            return_list (bool, optional): Whether to return a list of messages for the Jinja template to render. Defaults to False.
        
        Returns:
            Union[str, List[Dict[str, str]]]: A full string of conversation history or a list of messages for the Jinja template to render.

    `to_dict(self) ‑> Dict[str, Any]`
    :   Export the class as a dictionary.
        
        Returns:
            Dict[str, Any]: Prompt format as a dictionary.