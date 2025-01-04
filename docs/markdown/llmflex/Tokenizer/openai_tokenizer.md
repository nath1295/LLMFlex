Module llmflex.Tokenizer.openai_tokenizer
=========================================

Classes
-------

`OpenAITokenizer(model_id: str, **kwargs)`
:   OpenAI tokenizer class.
        
    
    Initialize the tokenizer with the given model ID.
    
    Args:
        model_id (str): The ID of the OpenAI model to use for tokenization.

    ### Ancestors (in MRO)

    * llmflex.Tokenizer.base_tokenizer.BaseTokenizer
    * abc.ABC

    ### Static methods

    `from_hf_tokenizer(tokenizer: Encoding)`
    :   Initialise the tokenizer from a tiktoken tokenizer directly.
        
        Args:
            tokenizer (Encoding): Tiktoken tokenizer.
        
        Returns:
            OpenAITokenizer: Initialised OpenAI tokenizer.

    ### Instance variables

    `openai_tokenizer: 'Encoding'`
    :   Returns the underlying tiktoken tokenizer instance.
        
        Returns:
            PreTrainedTokenizerBase: The underlying tiktoken tokenizer instance.