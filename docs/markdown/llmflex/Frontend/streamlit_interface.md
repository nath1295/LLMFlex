Module llmflex.Frontend.streamlit_interface
===========================================

Functions
---------

    
`create_streamlit_script(model_kwargs: Dict[str, Any], embeddings_kwargs: Dict[str, Any], tool_kwargs: List[Dict[str, Any]] = [], auth: Optional[Tuple[str, str]] = None, debug: bool = False) ‑> str`
:   Create the script to run the streamlit interface.
    
    Args:
        model_kwargs (Dict[str, Any]): Kwargs to initialise the LLM factory.
        embeddings_kwargs (Dict[str, Any]): Kwargs to initialise the embeddings toolkit.
        tool_kwargs (List[Dict[str, Any]], optional): List of kwargs to initialise the tools. Defaults to [].
        auth (Optional[Tuple[str, str]], optional): Tuple of username and password. Defaults to None.
        debug (bool, optional): Whether to display the debug buttons. Defaults to False.
    
    Returns:
        str: The streamlit script as a string.

    
`embeddings_loader(embeddings_kwargs: Dict[str, Any]) ‑> llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit`
:   Load the embeddings given the kwargs.
    
    Args:
        embeddings_kwargs (Dict[str, Any]): Kwargs to initialise the embeddings toolkit.
    
    Returns:
        BaseEmbeddingsToolkit: The embeddings toolkit.

    
`run_streamlit_interface(model_kwargs: Dict[str, Any], embeddings_kwargs: Dict[str, Any], tool_kwargs: List[Dict[str, Any]] = [], auth: Optional[Tuple[str, str]] = None, debug: bool = False, app_name: str = 'LLMFlex') ‑> None`
:   Run the streamlit interface.
    
    Args:
        model_kwargs (Dict[str, Any]): Kwargs to initialise the LLM factory.
        embeddings_kwargs (Dict[str, Any]): Kwargs to initialise the embeddings toolkit.
        tool_kwargs (List[Dict[str, Any]], optional): List of kwargs to initialise the tools. Defaults to [].
        auth (Optional[Tuple[str, str]], optional): Tuple of username and password. Defaults to None.
        debug (bool, optional): Whether to display the debug buttons. Defaults to False.
        app_name (str, optional): name of the streamlit script created. Defaults to PACKAGE_DISPLAY_NAME.

    
`tool_loader(tool_kwargs: Dict[str, Any], embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit], model: llmflex.Models.Factory.llm_factory.LlmFactory) ‑> llmflex.Tools.base_tool.BaseTool`
:   Load the embeddings given the kwargs.
    
    Args:
        tool_kwargs (Dict[str, Any]): Kwargs to initialise the tool.
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit for the tool if needed.
        model (LlmFactory): LlmFactory for the tool if needed.
    
    Returns:
        BaseTool: The tool.

Classes
-------

`InterfaceState(model: llmflex.Models.Factory.llm_factory.LlmFactory, embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit], tools: List[Type[llmflex.Tools.base_tool.BaseTool]] = [])`
:   Initialise the backend of the Streamlit interface.
    
    Args:
        model (LlmFactory): LLM factory.
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit.
        tools (List[Type[BaseTool]], optional): List of tools. Defaults to [].

    ### Instance variables

    `current_title: str`
    :   Current memory chat title.
        
        Returns:
            str: Current memory chat title.

    `history: List[List[str]]`
    :   Current conversation history.
        
        Returns:
            List[List[str]]: Current conversation history.

    `presets: List[str]`
    :   List of prompt templates presets.
        
        Returns:
            List[str]: List of prompt templates presets.

    `titles: List[str]`
    :   All existing chat titles.
        
        Returns:
            List[str]: All existing chat titles.

`StreamlitInterface(model_kwargs: Dict[str, Any], embeddings_kwargs: Dict[str, Any], tool_kwargs: List[Dict[str, Any]] = [], auth: Optional[Tuple[str, str]] = None, debug: bool = False)`
:   

    ### Instance variables

    `ai_start_text: str`
    :

    `backend: llmflex.Frontend.streamlit_interface.InterfaceState`
    :

    `conversation_delete: bool`
    :

    `experimental: bool`
    :

    `generating: bool`
    :   Whether chatbot is generating.

    `generation_config: Dict[str, str]`
    :

    `generation_time_info: str`
    :

    `history_dict: List[Dict[str, Any]]`
    :

    `islogin: bool`
    :

    `login_wrong: bool`
    :

    `mobile: bool`
    :

    `tool_states: Dict[str, bool]`
    :

    ### Methods

    `add_chat(self, title: str) ‑> None`
    :

    `ai_start_textbox(self) ‑> None`
    :

    `assistant_response(self, ex: Dict[str, Any], i: int, last: int) ‑> None`
    :

    `chatbot(self) ‑> None`
    :

    `conversation_history(self) ‑> None`
    :

    `conversations(self) ‑> None`
    :   List of conversations.

    `create_generation_config(self, gen_type: Literal['new', 'retry', 'continue', 'none'], user_input: str, ai_start: Optional[str] = None) ‑> None`
    :

    `current_prompt_index(self) ‑> int`
    :

    `delete_chat(self, title: str) ‑> None`
    :

    `experimental_buttons(self) ‑> None`
    :

    `get_generation_iterator(self) ‑> Iterator[str]`
    :

    `get_history(self, mem) ‑> List[Dict[str, Any]]`
    :

    `get_tool(self, user_input: str) ‑> Optional[Type[llmflex.Tools.base_tool.BaseTool]]`
    :

    `input_template(self, user: Optional[str], ai_start: str) ‑> Optional[Dict[str, Any]]`
    :

    `launch(self) ‑> None`
    :

    `llm_settings(self) ‑> None`
    :   LLM generation settings.

    `login(self) ‑> None`
    :

    `login_with_cred(self, user: str, password: str) ‑> None`
    :

    `memory_settings(self) ‑> None`
    :   Memory token limit settings.

    `new_chat_form(self) ‑> None`
    :

    `prompt_template_settings(self) ‑> None`
    :   Prompt template settings.

    `refresh_history(self) ‑> None`
    :

    `remove_last(self) ‑> None`
    :

    `retry_response(self, cont: bool = False, ai_start: str = '') ‑> None`
    :

    `run_tool(self, tool: llmflex.Tools.base_tool.BaseTool, user_input: str) ‑> Iterator[Union[str, Tuple[str, str], Iterator[str]]]`
    :

    `save_interaction(self, user: str, assistant: str, **kwargs) ‑> None`
    :

    `set_exeperimental(self) ‑> None`
    :

    `set_llm_config(self, temperature: float, max_new_tokens: int, repetition_penalty: float, top_p: float, top_k: int) ‑> None`
    :

    `set_memory_settings(self, long: int, short: int, score: float) ‑> None`
    :

    `set_mobile(self) ‑> None`
    :

    `set_prompt_template(self) ‑> None`
    :

    `set_system_message(self, system: str) ‑> None`
    :

    `set_time_info(self, time: Union[str, float] = '--') ‑> None`
    :

    `settings(self) ‑> None`
    :   Settings of the webapp.

    `sidebar(self) ‑> None`
    :   Sidebar of the webapp.

    `switch_chat(self, title: str) ‑> None`
    :

    `system_prompt_settings(self) ‑> None`
    :   System prompt settings.

    `test_buttons(self) ‑> None`
    :

    `toggle_conversation_delete(self) ‑> None`
    :

    `toggle_exeperimental(self) ‑> None`
    :

    `toggle_generating(self) ‑> None`
    :

    `toggle_mobile(self) ‑> None`
    :

    `toggle_tool(self, tool_name: str) ‑> None`
    :

    `tool_settings(self) ‑> None`
    :

    `user_input_box(self) ‑> None`
    :