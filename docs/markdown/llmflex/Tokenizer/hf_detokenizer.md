Module llmflex.Tokenizer.hf_detokenizer
=======================================

Functions
---------

`get_detokenizer(pretrained_model_name_or_path: str, tokenizer: PreTrainedTokenizer) ‑> llmflex.Tokenizer.hf_detokenizer.NaiveDetokenizer | llmflex.Tokenizer.hf_detokenizer.SPMDetokenizer`
:   Get the detokenizer.
    
    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.

Classes
-------

`NaiveDetokenizer(tokenizer: PreTrainedTokenizer)`
:   

    ### Instance variables

    `last_segments: List[str]`
    :

    ### Methods

    `add_tokens(self, token_ids: List[List[int]]) ‑> None`
    :

    `finalize(self) ‑> None`
    :

    `reset(self, num_seqs: int | None = None) ‑> None`
    :

`SPMDetokenizer(tokenizer: PreTrainedTokenizer, trim_space=True)`
:   A streaming detokenizer for SPM models.
    
    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.

    ### Instance variables

    `last_segments: List[str]`
    :   Return the last segment of readable text since last time this property was accessed.

    ### Methods

    `add_tokens(self, token_ids: List[List[int]]) ‑> None`
    :

    `finalize(self)`
    :

    `reset(self, num_seqs: int | None = None) ‑> None`
    :