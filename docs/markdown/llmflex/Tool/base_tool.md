Module llmflex.Tool.base_tool
=============================

Functions
---------

`format_tool_name(s: str) ‑> str`
:   Format the tool class name into a lower class string.
    
    Args:
        s (str): The tool class name to be formatted.
    Returns:
        str: The formatted tool class name in lower case.

Classes
-------

`BaseTool()`
:   Base class for tools.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.Tool.base_tool.FunctionTool
    * llmflex.Tool.browser_tool.BrowserTool

    ### Instance variables

    `name: str`
    :   Get the name of the tool.
        
        This property returns the name of the tool, which is derived from the class name.
        
        Returns:
            str: The name of the tool.

    ### Methods

    `run(self, **kwargs) ‑> llmflex.Tool.base_tool.ToolOutput`
    :   Run the tool and structure the output into a ToolOutput object. Re-implement it for the subclass if necessary.
        
        Returns:
            ToolOutput: Structured tool output.

`FunctionTool(fn: Callable)`
:   Tool wrapper for python functions.
        
    
    Initialise the function as a tool.
    
    Args:
        fn (Callable): The python function.

    ### Ancestors (in MRO)

    * llmflex.Tool.base_tool.BaseTool
    * abc.ABC

`ToolOutput(**data: Any)`
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

    `content: str`
    :

    `footnotes: List[str] | None`
    :

    `images: List[str] | None`
    :

    `model_computed_fields`
    :

    `model_config`
    :

    `model_fields`
    :