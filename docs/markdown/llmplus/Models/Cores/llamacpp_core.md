Module llmplus.Models.Cores.llamacpp_core
=========================================

Functions
---------

    
`get_model_dir(model_id: str, model_file: Optional[str] = None) ‑> str`
:   Download the model file from Huggingface and get the local directory.
    
    Args:
        model_id (str): Model's HuggingFace ID.
        model_file (Optional[str], optional): Specific model quant file. If None, will choose the smallest quant automatically. Defaults to None.
    
    Returns:
        str: Local directory of the model file.

Classes
-------

`LlamaCppCore(model_id_or_path: str, model_file: Optional[str] = None, context_length: int = 4096, from_local: bool = False, **kwargs)`
:   This is the core class of loading model in gguf format.
        
    
    Initialising the core.
    
    Args:
        model_id (str): Model id (from Huggingface) or model file path to use.
        model_file (Optional[str], optional): Specific GGUF model to use. If None, the lowest quant will be used. Defaults to None.
        context_length (int, optional): Context length of the model. Defaults to 4096.
        from_local (bool, optional): Whether to treat the model_id given as a local path or a Huggingface ID. Defaults to False.

    ### Ancestors (in MRO)

    * llmplus.Models.Cores.base_core.BaseCore
    * abc.ABC