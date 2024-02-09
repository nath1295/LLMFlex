Module llmplus.Frontend.chat_interface
======================================

Classes
-------

`ChatInterface(model: llmplus.Models.Factory.llm_factory.LlmFactory, embeddings: Type[llmplus.Embeddings.base_embeddings.BaseEmbeddingsToolkit])`
:   

    ### Instance variables

    `buttons: List[str]`
    :   List of buttons except send.
        
        Returns:
            List[str]: List of buttons except send.

    `config_dict: Dict[str, Any]`
    :

    `current_title: str`
    :   Current memory chat title.
        
        Returns:
            str: Current memory chat title.

    `history: List[List[str]]`
    :   Current conversation history.
        
        Returns:
            List[List[str]]: Current conversation history.

    `mobile_config_dict: Dict[str, Any]`
    :

    `presets: List[str]`
    :   List of prompt templates presets.
        
        Returns:
            List[str]: List of prompt templates presets.

    `titles: List[str]`
    :   All existing chat titles.
        
        Returns:
            List[str]: All existing chat titles.

    ### Methods

    `change_llm_setting(self, temperature: float, max_tokens: int, repeat_penalty: float, top_p: float, top_k: int) ‑> str`
    :   Change llm generation settings.
        
        Args:
            temperature (float): Temperature of the llm.
            max_tokens (int): Maximum number of tokens to generate.
            repeat_penalty (float): Repetition penalty.
            top_p (float): Top P.
            top_k (int): Top K.

    `change_memory(self, btn: str, title: str) ‑> Tuple[Any]`
    :   Handling memory changing settings.
        
        Args:
            btn (str): Button that triggered this function.
            title (str): Title used for this trigger.
        
        Returns:
            Tuple[Any]: New title textbox, Chats dropdown menu, the Chatbot box, and the system textbox.

    `change_memory_setting(self, system: str, long_limit: int, short_limit: int, sim_score: float) ‑> str`
    :   Changing the system message and memory settings.
        
        Args:
            system (str): System textbox.
            long_limit (int): Long term memory slider.
            short_limit (int): Short term memory slider.
            sim_score (float): Similarity score threshold slider.

    `change_prompt_format(self, template: str) ‑> str`
    :   Changing the prompt format.
        
        Args:
            template (str): Preset name from the dropdown menu.

    `generation(self, bot: List[List[str]]) ‑> List[List[str]]`
    :   Text generation.
        
        Args:
            bot (List[List[str]]): Chatbot conversation history.
        
        Returns:
            List[List[str]]: The updated conversation.

    `get_llm_settings(self) ‑> str`
    :

    `get_memory_settings(self) ‑> str`
    :

    `get_prompt(self, user_input: str, ai_start: str = '') ‑> str`
    :

    `get_prompt_settings(self) ‑> str`
    :

    `input_handler(self, btn: str, user: str, start: str, bot: List[List[str]]) ‑> Tuple[Any]`
    :   Handling GUI and class attributes before text generation.
        
        Args:
            btn (str): Button used to trigger the function.
            user (str): User input.
            start (str): Start of the chatbot output, should be an empty string by default.
            bot (List[List[str]]): Chatbot conversation history.
        
        Returns:
            Tuple[Any]: send button, user input box, conversation box, and all other buttons.

    `launch(self, mobile: bool = False, **kwargs) ‑> None`
    :

    `output_map(self, keys: Union[str, List[str]]) ‑> List[Any]`
    :

    `postgen_handler(self) ‑> List[Any]`
    :   Reactivate all the buttons.
        
        Returns:
            List[Any]: All buttons.

    `remove_last(self) ‑> List[List[str]]`
    :   Removing the last interaction of the conversation.
        
        Returns:
            List[List[str]]: Conversation history after removing the last interaction.

    `vars(self, var_name: str, **kwargs: Dict[str, Any]) ‑> Any`
    :   Generate the gradio component in the config dictionary given the key in the config dict.
        
        Args:
            var_name (str): Key in the config dict.
        
        Returns:
            Any: The gradio component.