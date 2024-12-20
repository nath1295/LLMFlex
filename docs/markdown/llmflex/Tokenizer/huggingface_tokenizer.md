Module llmflex.Tokenizer.huggingface_tokenizer
==============================================

Classes
-------

`HuggingFaceTokenizer(pretrained_model_name_or_path: str, **kwargs)`
:   Huggingface tokenizer class.
        
    
    Initialise the tokenizer. For initialisation kwargs, please pass in other keyword arguments.
    
    Args:
        pretrained_model_name_or_path (str): Huggingface repository name or path to the model.

    ### Ancestors (in MRO)

    * llmflex.Tokenizer.base_tokenizer.BaseTokenizer
    * abc.ABC

    ### Static methods

    `from_hf_tokenizer(tokenizer: PreTrainedTokenizerBase) ‑> llmflex.Tokenizer.huggingface_tokenizer.HuggingFaceTokenizer`
    :   Initialise the tokenizer from a huggingface transformers tokenizer directly.
        
        Args:
            tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer.
        
        Returns:
            HuggingFaceTokenizer: Initialised huggingface tokenizer.

    ### Instance variables

    `hf_tokenizer: PreTrainedTokenizerBase`
    :   Returns the underlying HuggingFace tokenizer instance.
        
        Returns:
            PreTrainedTokenizerBase: The underlying HuggingFace tokenizer instance.