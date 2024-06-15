Module llmflex.Frontend.streamlit_interface
===========================================

Functions
---------

    
`create_streamlit_script() ‑> str`
:   Create the main script to run the app.
    
    Returns:
        str: Script directory.

    
`get_backend() ‑> llmflex.Frontend.app_resource.AppBackend`
:   Get the backend object for the webapp.
    
    Returns:
        AppBackend: Backend object for the webapp.

Classes
-------

`AppInterface()`
:   

    ### Instance variables

    `backend: llmflex.Frontend.app_resource.AppBackend`
    :   Backend resources.
        
        Returns:
            AppBackend: Backend resources.

    `begin_text_cache: str`
    :   Begin text cache.
        
        Returns:
            str: Begin text cache.

    `chat_delete_button: bool`
    :   Whether to show chat deletion buttons.
        
        Returns:
            bool: Whether to show chat deletion buttons.

    `credentials: Optional[Tuple[str, str]]`
    :   Login credentials if provided.
        
        Returns:
            Optional[Tuple[str, str]]: Login credentials if provided.

    `enable_begin_text: bool`
    :   Whether on mobile device.
        
        Returns:
            bool: Whether on mobile device.

    `generating: bool`
    :   Whether text generation in progress.
        
        Returns:
            bool: Whether text generation in progress.

    `input_config: Optional[Dict[str, Any]]`
    :   Configuration for text generation if available.
        
        Returns:
            Optional[Dict[str, Any]]: Configuration for text generation if available.

    `is_login: bool`
    :   Whether it is logged in or not.
        
        Returns:
            bool: Whether it is logged in or not.

    `kb_create_button: bool`
    :   Whether to show knowledge base creation buttons.
        
        Returns:
            bool: Whether to show knowledge base creation buttons.

    `kb_delete_button: bool`
    :   Whether to show knowledge base deletion buttons.
        
        Returns:
            bool: Whether to show knowledge base deletion buttons.

    `mobile: bool`
    :   Whether on mobile device.
        
        Returns:
            bool: Whether on mobile device.

    ### Methods

    `begin_text_settings(self) ‑> None`
    :   Create toggle button for response starting text.

    `chatbot(self) ‑> None`
    :   Creating the chatbot interface.

    `chats(self) ‑> None`
    :   Listing all conversations.

    `create_input_config(self, user_input: str, begin_text: str, generation_mode: Literal['new', 'retry', 'continue']) ‑> None`
    :   Create everything needed for the next generation.
        
        Args:
            user_input (str): User request.
            begin_text (str): Starting text of the response.
            generation_mode (Literal[&#39;new&#39;, &#39;retry&#39;, &#39;continue&#39;]): Mode of generation.

    `generation_message(self) ‑> None`
    :   Create streaming message.

    `historical_conversation(self) ‑> None`
    :   Creating the historical conversation.

    `input_box(self) ‑> None`
    :   Creating the input box.

    `knowledge_base_config(self) ‑> None`
    :   Creating knowledge base configurations.

    `knowledge_base_creation(self) ‑> None`
    :   Creating knowledge base.

    `knowledge_base_settings(self) ‑> None`
    :   Create settings for memory.

    `login(self) ‑> None`
    :   Creating login page.

    `memory_settings(self) ‑> None`
    :   Create settings for memory.

    `mobile_settings(self) ‑> None`
    :   Create toggle button for mobile mode.

    `model_settings(self) ‑> None`
    :   Create settings for text generation.

    `process_footnote(self, tool_output: Dict[str, Any]) ‑> str`
    :   Check if footnote exist in the tool output.
        
        Args:
            tool_output (Dict[str, Any]): Tool output dictionary.
        
        Returns:
            str: If footnote exist, return the footnote string, otherwise return a empty string.

    `process_image(self, messages: List[Dict[str, Any]], tool_output: Dict[str, Any]) ‑> str`
    :   Capture images in tool output.
        
        Args:
            messages (List[Dict[str, Any]]): List of messages to form the prompt.
            tool_output (Dict[str, Any]): Output of the tool.
        
        Returns:
            str: Prompt for generation.

    `prompt_format_settings(self) ‑> None`
    :   Prompt format settings.

    `run(self) ‑> None`
    :   Run the app.

    `settings(self) ‑> None`
    :   Creating the settings.

    `sidebar(self) ‑> None`
    :   Creating the sidebar.

    `system_message_setttings(self) ‑> None`
    :   Create settings for system message.

    `tool_settings(self) ‑> None`
    :   Create settings for tools.

    `util_buttons(self) ‑> None`
    :   Create utility buttons for text generation.