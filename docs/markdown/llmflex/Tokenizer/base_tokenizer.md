Module llmflex.Tokenizer.base_tokenizer
=======================================

Classes
-------

`BaseTokenizer(tokenizer_type: str, eos_token: str | None, bos_token: str | None, pad_token: str | None, eos_token_id: int | None, bos_token_id: int | None, pad_token_id: int | None)`
:   Base class for tokenizers.
        
    
    Initialize the tokenizer.
    
    Args:
        tokenizer_type (str): The type of the tokenizer.
        eos_token (Optional[str]): The end-of-sequence token if the tokenizer is for an LLM model.
        bos_token (Optional[str]): The beginning-of-sequence token if the tokenizer is for an LLM model.
        pad_token (Optional[str]): The padding token for the tokenizer.
        eos_token_id (Optional[int]): The end-of-sequence token ID if the tokenizer is for an LLM model.
        bos_token_id (Optional[int]): The beginning-of-sequence token ID if the tokenizer is for an LLM model.
        pad_token_id (Optional[int]): The padding token ID for the tokenizer.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.Tokenizer.huggingface_tokenizer.HuggingFaceTokenizer
    * llmflex.Tokenizer.llamacpp_tokenizer.LlamaCppTokenizer
    * llmflex.Tokenizer.openai_tokenizer.OpenAITokenizer

    ### Instance variables

    `bos_token: str | None`
    :   BOS token text.
        
        Returns:
            Optional[str]: BOS token text.

    `bos_token_id: int | None`
    :   BOS token id.
        
        Returns:
            Optional[int]: BOS token id.

    `eos_token: str | None`
    :   EOS token text.
        
        Returns:
            Optional[str]: EOS token text.

    `eos_token_id: int | None`
    :   EOS token id.
        
        Returns:
            Optional[int]: EOS token id.

    `pad_token: str | None`
    :   Pad token text.
        
        Returns:
            Optional[str]: Pad token text.

    `pad_token_id: int | None`
    :   Pad token id.
        
        Returns:
            Optional[int]: Pad token id.

    `tokenizer_type: str`
    :   Class type of tokenizer.
        
        Returns:
            str: Class type of tokenizer.

    ### Methods

    `batch_detokenize(self, token_ids: List[List[int]], skip_special_tokens: bool = True) ‑> List[str]`
    :   Detokenize a batch of token IDs.
        Args:
            token_ids (List[List[int]]): List of token IDs to detokenize.
            skip_special_tokens (bool, optional): Whether to skip special tokens (BOS, EOS, PAD) during detokenization. Defaults to True.
        
        Returns:
            List[str]: List of detokenized texts for each list of token IDs in the input list.

    `batch_tokenize(self, texts: List[str], add_special_tokens: bool = True) ‑> List[List[int]]`
    :   Tokenize a batch of texts.
        Args:
            texts (List[str]): List of texts to tokenize.
            add_special_tokens (bool, optional): Whether to add special tokens (BOS, EOS, PAD) to the tokenized texts. Defaults to True.
        
        Returns:
            List[List[int]]: List of token IDs for each text in the input list.

    `detokenize(self, token_ids: List[int], skip_special_tokens: bool = True) ‑> str`
    :   Decode a list of token ids to the original text.
        
        Args:
            token_ids (List[int]): List of token IDs.
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.
        
        Returns:
            str: Decoded text.

    `get_num_tokens(self, text: str, add_special_tokens: bool = True) ‑> int`
    :   Get the number of tokens.
        
        Args:
            text (str): String to count tokens.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.
        
        Returns:
            int: Number of tokens.

    `tokenize(self, text: str, add_special_tokens: bool = True) ‑> List[int]`
    :   Tokenize a string.
        
        Args:
            text (str): String to tokenize.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.
        
        Returns:
            List[int]: List of token IDs.