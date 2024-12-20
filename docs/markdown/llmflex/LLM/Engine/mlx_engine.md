Module llmflex.LLM.Engine.mlx_engine
====================================

Classes
-------

`MLXEngine(pretrained_model_name_or_path: str, quantization: Literal['fp16', 'q8', 'q4', 'q2'] = 'fp16', tokenizer_name_or_path: str | None = None, revision: str | None = None, model_kwargs: Dict[str, Any] | None = None, tokenizer_kwargs: Dict[str, Any] | None = None, prefill_step_size: int = 512, **kwargs)`
:   Class for LLM engine using MLX.
        
    
    Initializes the MLXEngine with the given parameters.
    
    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
        quantization (Literal['fp16', 'q8', 'q4', 'q2'], optional): The quantization method to use for the model. Defaults to 'fp16'.
        tokenizer_name_or_path (Optional[str], optional): The name or path of the tokenizer to use. Defaults to None.
        revision (Optional[str], optional): The branch of the Huggingface repository to use. Defaults to None.
        model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the model. Defaults to None.
        tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
        prefill_step_size (int, optional): The batch size of prompt processing. Defaults to 512.

    ### Ancestors (in MRO)

    * llmflex.LLM.Engine.base_engine.BaseEngine
    * abc.ABC