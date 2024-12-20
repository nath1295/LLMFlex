Module llmflex.Schema.document
==============================

Classes
-------

`Document(**data: Any)`
:   Class for texts with metadata.
        
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `metadata: Dict[str, Any]`
    :

    `model_computed_fields`
    :

    `model_config`
    :

    `model_fields`
    :

    `text: str`
    :