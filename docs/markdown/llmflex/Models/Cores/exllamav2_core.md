Module llmflex.Models.Cores.exllamav2_core
==========================================

Functions
---------

    
`get_exl2_model_dir(repo_id: str, revision: Optional[str] = None) ‑> str`
:   Download and get the model repository local directory.
    
    Args:
        repo_id (str): Huggingface model ID.
        revision (Optional[str], optional): Branch of the repository. If None is given, the main branch will be used. Defaults to None.
    
    Returns:
        str: Model local directory.

Classes
-------

`Exl2Core(repo_id: str, revision: Optional[str] = None, **kwargs)`
:   Base class of Core object to store the llm model and tokenizer.
        
    
    Initialise the exl2 model core.
    
    Args:
        repo_id (str): Huggingface model ID.
        revision (Optional[str], optional): Branch of the repository. If None is given, the main branch will be used. Defaults to None.

    ### Ancestors (in MRO)

    * llmflex.Models.Cores.base_core.BaseCore
    * abc.ABC

    ### Methods

    `decode(self, token_ids: List[int]) ‑> str`
    :   Untokenize a list of tokens.
        
        Args:
            token_ids (List[int]): Token ids to untokenize.
        
        Returns:
            str: Untokenized string.