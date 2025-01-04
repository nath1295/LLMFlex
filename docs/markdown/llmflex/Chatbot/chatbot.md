Module llmflex.Chatbot.chatbot
==============================

Functions
---------

`direct_response() ‑> None`
:   If no tool is required to respond to the user, use this tool to respond directly.

Classes
-------

`Chatbot(llm: BaseLLM, memory: ForwardRef('BaseMemory') | None = None, tools: List[ForwardRef('BaseTool') | Callable] | None = None, chat_template: ForwardRef('ChatTemplate') | None = None)`
:   Class providing chatbot functionalities with an LLM.
        
    
    Initializes the chatbot with the provided LLM, memory, tools, and chat template.
    
    Args:
        llm (BaseLLM): The language model engine to use for generating responses.
        memory (Optional[BaseMemory], optional): The memory system to use for storing and retrieving conversation history. If not provided, a `ChatMemory` instance will be initiated. Defaults to None.
        tools (Optional[List[Union[BaseTool, Callable]]], optional): A list of tools that the chatbot can use to assist with responses. All turned off by default. Defaults to None.
        chat_template (Optional[ChatTemplate], optional): The chat template to use for formatting the conversation. If not provided, the default LLM chat template will be used. Defaults to None.

    ### Instance variables

    `chat_template: ChatTemplate`
    :   Chat template for formatting conversation.
        
        Returns:
            ChatTemplate: Chat template for formatting conversation.

    `generation_kwargs: Dict[str, Any]`
    :   Generation configuration of the LLM.
        
        Returns:
            Dict[str, Any]: Generation configuration of the LLM.

    `history: List[Dict[str, Any]]`
    :   Conversation history.
        
        Returns:
            List[Dict[str, Any]]: Conversation history.

    `llm: BaseLLM`
    :   LLM for the chatbot.
        
        Returns:
            BaseLLM: LLM for the chatbot.

    `memory: BaseMemory`
    :   Conversation memory.
        
        Returns:
            BaseMemory: Conversation memory.

    `memory_config: llmflex.Chatbot.chatbot.MemoryConfig`
    :   Memory configuration.
        
        Returns:
            MemoryConfig: Memory configuration.

    `system_message: str | None`
    :   System message of the conversation.
        
        Returns:
            Optional[str]: System message of the conversation.

    `title: str | None`
    :   Title of the conversation if set.
        
        Returns:
            Optional[str]: Title of the conversation if set.

    `tool_dict: Dict[str, bool]`
    :   Dictionary of tools with keys as the tool names and values as whether the tool is turned on or not.
        
        Returns:
            Dict[str, bool]: Dictionary of tools with keys as the tool names and values as whether the tool is turned on or not.

    `tools: List[BaseTool]`
    :   List of available tools.
        
        Returns:
            List[BaseTool]: List of available tools.

    ### Methods

    `chat(self, user_message: str, stream: bool = False) ‑> str | Tuple[str, Dict[str, Any]] | Iterator[Dict[str, Any] | str]`
    :   Generate a response to the user's message.
        
        Args:
            user_message (str): The user's message to be processed.
            stream (bool, optional): Whether to stream the response. Defaults to False.
        
        Returns:
            Union[str, Tuple[str, Dict[str, Any]], Iterator[Union[Dict[str, Any], str]]]: The response to the user's message. A dictionary will be yielded for tool calls.

    `notebook_chat(self, user_message: str) ‑> None`
    :   Chating with the bot in a Jupyter notebook, with conversation history formatted and displayed as markdown. 
        
        Args:
            user_message (str): The new user message.

    `prepare_prompt(self, user_message: str, tool_call: Dict[str, Any] | None = None) ‑> str`
    :   Prepare the prompt for generation.
        
            user_message (str): The user's message to be processed.
            tool_call (Optional[Dict[str, Any]], optional): A dictionary containing information about a tool call. Defaults to None.
        
        Returns:
            str: The prompt to be passed to the LLM.

    `remove_last_interaction(self, keep_last_user: bool = False) ‑> None`
    :   Remove the last interaction from the memory.
        
        Args:
            keep_last_user (bool, optional): Whether to keep the last user message. Defaults to False.

    `set_generation_kwargs(self, **kwargs) ‑> None`
    :   Set the generation configuration of the LLM by passing keyword arguments.

    `set_memory_config(self, recent_token_limit: int | None = None, relevant_token_limit: int | None = None, relevance_score_threshold: float | None = None) ‑> None`
    :   Set the memory configuration.
        
        Args:
            recent_token_limit (Optional[int], optional): The maximum number of tokens to retrieve from the latest conversation turns of the memory. Defaults to None.
            relevant_token_limit (Optional[int], optional): The maximum number of tokens to retrieve from the older but relevant conversation turns of the memory.. Defaults to None.
            relevance_score_threshold (Optional[float], optional): The threshold for relevance score. Defaults to None.

    `set_system_message(self, system_message: str) ‑> None`
    :   Set the system message of the conversation.
        
        Args:
            system_message (str): The new system message of the conversation.

    `set_title(self, title: str | None) ‑> None`
    :   Set the title of the conversation.
        
        Args:
            title (Optional[str]): The new title of the conversation.

    `toggle_tool(self, tools: str | List[str]) ‑> None`
    :   Toggle the given tool(s) for their availability of the chatbot.
        
        Args:
            tools (Union[str, List[str]]): List of tools to toggle.

`MemoryConfig(**data: Any)`
:   Usage docs: https://docs.pydantic.dev/2.7/concepts/models/
    
    A base class for creating Pydantic models.
    
    Attributes:
        __class_vars__: The names of classvars defined on the model.
        __private_attributes__: Metadata about the private attributes of the model.
        __signature__: The signature for instantiating the model.
    
        __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
        __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
        __pydantic_custom_init__: Whether the model has a custom `__init__` function.
        __pydantic_decorators__: Metadata containing the decorators defined on the model.
            This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
        __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
            __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
        __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
        __pydantic_post_init__: The name of the post-init method for the model, if defined.
        __pydantic_root_model__: Whether the model is a `RootModel`.
        __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the model.
        __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the model.
    
        __pydantic_extra__: An instance attribute with the values of extra fields from validation when
            `model_config['extra'] == 'allow'`.
        __pydantic_fields_set__: An instance attribute with the names of fields explicitly set.
        __pydantic_private__: Instance attribute with the values of private attributes set on the model instance.
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `model_computed_fields`
    :

    `model_config`
    :

    `model_fields`
    :

    `recent_token_limit: int`
    :

    `relevance_score_threshold: float`
    :

    `relevant_token_limit: int`
    :