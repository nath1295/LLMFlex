Module llmflex.Prompt.chat_template
===================================

Functions
---------

`get_chat_template(chat_template: Literal['chatml', 'llama3', 'mistral', 'gemma', 'deepseek', 'openchat', 'phi'] | None = None, tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer | None = None, tools: List[Dict[str, Any]] | None = None) ‑> str`
:   Retrieve the chat template based on the provided key, tokenizer, and tool availability.
    
    This function returns the appropriate chat template based on the given `chat_template` key, `tokenizer`, and the existence of `tools`.
    
    Args:
        chat_template (Optional[CHAT_TEMPLATE_PRESETS], optional): The key to the chat template to use. Defaults to None.
        tokenizer (Optional[BaseTokenizer], optional): The tokenizer that provides information about the chat template. Defaults to None.
        tools (Optional[List[Dict[str, Any]]], optional): The list of tools to use in the template. Defaults to None.
    
    Returns:
        str: The Jinja chat template.

Classes
-------

`ChatMessage(**data: Any)`
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

    `content: Any | None`
    :

    `model_computed_fields`
    :

    `model_config`
    :

    `model_fields`
    :

    `role: Literal['system', 'user', 'assistant', 'tool']`
    :

    `tool_calls: List[llmflex.Prompt.chat_template.ToolCall] | None`
    :

`ChatTemplate(tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer, chat_template: Literal['chatml', 'llama3', 'mistral', 'gemma', 'deepseek', 'openchat', 'phi'] | None = None)`
:   A class to manage chat templates for conversational AI.
    
    This class provides a way to retrieve chat templates based on the given chat template key, tokenizer, and the existence of tools.
    It also handles the case where the tokenizer is a Hugging Face tokenizer and extracts the appropriate chat template from its configuration.
    
    Initialize the ChatTemplate instance with the provided tokenizer and chat template key.
    
    Args:
        tokenizer (BaseTokenizer): The tokenizer that provides information about the chat template.
        chat_template (Optional[CHAT_TEMPLATE_PRESETS], optional): The key to the chat template to use. Defaults to None.

    ### Instance variables

    `allow_multiple_assistant: bool`
    :   Checks if the chat template allows multiple consecutive assistant messages.
        
        Returns:
            bool: True if the chat template allows multiple consecutive assistant messages, False otherwise.

    `support_system: bool`
    :   Checks if the chat template supports system message.
        
        Returns:
            bool: True if the chat template supports system message, False otherwise.

    `support_tool_call: bool`
    :   Checks if the chat template supports tool calls.
        
        Returns:
            bool: True if the chat template supports tool calls, False otherwise.

    `tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer`
    :   Gets the tokenizer associated with this chat template.
        
        Returns:
            BaseTokenizer: The tokenizer associated with this chat template.

    `tool_start: str`
    :   Gets the start string for tool calls in the chat template.
        
        Returns:
            str: The start string for tool calls in the chat template.

    ### Methods

    `apply_chat_template(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None, tool_choice: Literal['none', 'auto', 'required'] | Dict[str, str | Dict[str, str]] = 'auto', add_generation_prompt: bool = True, continue_final_message: bool = False) ‑> str`
    :   Applies the chat template to the given messages and tools, considering tool choice.
        
        This method takes a list of messages, an optional list of tools, and a tool choice parameter, and applies the chat template to generate a string
        representing the conversation. If tools are provided, the method will include the information about the tools in the system prompt.
        The `tool_choice` parameter determines how tools are used in the conversation. If set to 'none', no tools will be used. If set to 'auto',
        tools will be used if the assistant's response includes a tool call. If set to 'required', at least one tool must be used in the conversation.
        If `add_generation_prompt` is set to True, the method will add a generation prompt to the end of the conversation string.
        
        Args:
            messages (List[Dict[str, Any]]): The list of messages to apply the chat template to.
            tools (Optional[List[Dict[str, Any]]], optional): The list of tools to include in the conversation. Defaults to None.
            tool_choice (Union[Literal['none', 'auto', 'required'], Dict[str, Union[str, Dict[str, str]]]], optional): The tool choice parameter. Defaults to 'auto'.
            add_generation_prompt (bool, optional): Whether to add a generation prompt to the end of the conversation string. Defaults to True.
            continue_final_message (bool, optional): Whether to continue the final message in the conversation string. Defaults to False.
        
        Returns:
            str: The conversation string generated by applying the chat template to the given messages and tools, considering tool choice.

`ToolCall(**data: Any)`
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

    `function: llmflex.Prompt.chat_template.ToolCallContent`
    :

    `model_computed_fields`
    :

    `model_config`
    :

    `model_fields`
    :

`ToolCallContent(**data: Any)`
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

    `arguments: str`
    :

    `model_computed_fields`
    :

    `model_config`
    :

    `model_fields`
    :

    `name: str`
    :