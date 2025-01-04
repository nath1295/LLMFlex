Module llmflex.Tokenizer.llamacpp_tokenizer
===========================================

Classes
-------

`LlamaCppTokenizer(pretrained_model_name_or_path: str, model_file: Optional[str] = None, **kwargs)`
:   Llama CPP tokenizer class.
        
    
    Initialise the tokenizer.
    
    Args:
        pretrained_model_name_or_path (str): Huggingface repository name or full path of the model of the model file.
        model_file (str, optional): Model filename if a HuggingFace repository name is given for tokenizer_id_or_path. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Tokenizer.base_tokenizer.BaseTokenizer
    * abc.ABC

    ### Static methods

    `from_llama_model(llama_model: "'Llama'")`
    :   Initialise the tokenizer from a Llama CPP model directly.
        
        Args:
            llama_model (Llama): Llama CPP model.
        
        Returns:
            LlamaCppTokenizer: Initialised Llama CPP tokenizer.

    ### Instance variables

    `llama_tokenizer: 'Llama'`
    :   Get the underlying Llama tokenizer.
        
        Returns:
            Llama: The Llama tokenizer.