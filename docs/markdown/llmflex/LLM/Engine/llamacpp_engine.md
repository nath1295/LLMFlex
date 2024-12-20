Module llmflex.LLM.Engine.llamacpp_engine
=========================================

Classes
-------

`LlamaCppEngine(pretrained_model_name_or_path: str, model_file: str | None = None, tokenizer_name_or_path: str | None = None, context_size: int = 8192, verbose: bool = False, download_kwargs: Dict[str, Any] | None = None, **kwargs)`
:   Class for LLM engine using llama-cpp-python.
        
    
    Initializes the LlamaCppEngine with the given parameters.
    
    Args:
        pretrained_model_name_or_path (str): The HuggingFace repository name or the full path of the model file.
        model_file (str, optional): The model filename if a HuggingFace repository name is given for tokenizer_id_or_path. Defaults to None.
        tokenizer_name_or_path (str, optional): The Huggingface repository name or the full path to load the HuggingFace tokenizer. If not provided, structured text generation might encounter problems. Defaults to None.
        context_size (int, optional): The size of the context window. Defaults to 8192.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        download_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the download function. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the Llama class.

    ### Ancestors (in MRO)

    * llmflex.LLM.Engine.base_engine.BaseEngine
    * abc.ABC