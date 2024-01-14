Module llmplus.Models.Cores.base_core
=====================================

Classes
-------

`BaseCore(model_id: str = 'gpt2', **kwargs)`
:   Base class of Core object to store the llm model and tokenizer.
        
    
    Initialising the core instance.
    
    Args:
        model_id (str, optional): Model id (from Huggingface) to use. Defaults to 'gpt2'.

    ### Descendants

    * llmplus.Models.Cores.huggingface_core.HuggingfaceCore
    * llmplus.Models.Cores.llamacpp_core.LlamaCppCore
    * llmplus.Models.Cores.openai_core.OpenAICore

    ### Instance variables

    `core_type: str`
    :   Type of core.
        
        Returns:
            str: Type of core.

    `model: Any`
    :   Model for llms.
        
        Returns:
            Any: Model for llms.

    `model_id: str`
    :   Model ID.
        
        Returns:
            str: Model ID.

    `prompt_template: llmplus.Prompts.prompt_template.PromptTemplate`
    :   Default prompt template for the model.
        
        Returns:
            PromptTemplate: Default prompt template for the model.

    `tokenizer: Any`
    :   Tokenizer of the model.
        
        Returns:
            Any: Tokenizer of the model.

    ### Methods

    `decode(self, token_ids: List[int]) ‑> str`
    :   Untokenize a list of tokens.
        
        Args:
            token_ids (List[int]): Token ids to untokenize. 
        
        Returns:
            str: Untokenized string.

    `encode(self, text: str) ‑> List[int]`
    :   Tokenize the given text.
        
        Args:
            text (str): Text to tokenize.
        
        Returns:
            List[int]: List of token ids.

    `unload(self) ‑> None`
    :   Unload the model from ram.

`BaseLLM(core: Type[llmplus.Models.Cores.base_core.BaseCore], generation_config: Dict[str, Any], stop: List[str])`
:   Base LLM class for llmplus, using the LLM class from langchain.
        
    
    Initialising the LLM.
    
    Args:
        core (Type[BaseCore]): The LLM model core.
        generation_config (Dict[str, Any]): Generation configuration.
        stop (List[str]): List of strings to stop the generation of the llm.

    ### Ancestors (in MRO)

    * langchain_core.language_models.llms.LLM
    * langchain_core.language_models.llms.BaseLLM
    * langchain_core.language_models.base.BaseLanguageModel
    * langchain_core.runnables.base.RunnableSerializable
    * langchain_core.load.serializable.Serializable
    * pydantic.v1.main.BaseModel
    * pydantic.v1.utils.Representation
    * langchain_core.runnables.base.Runnable
    * typing.Generic
    * abc.ABC

    ### Descendants

    * llmplus.Models.Cores.base_core.DebugLLM
    * llmplus.Models.Cores.huggingface_core.HuggingfaceLLM
    * llmplus.Models.Cores.llamacpp_core.LlamaCppLLM
    * llmplus.Models.Cores.openai_core.OpenAILLM

    ### Class variables

    `core: Type[llmplus.Models.Cores.base_core.BaseCore]`
    :

    `generation_config: Dict[str, Any]`
    :

    `stop: List[str]`
    :

    ### Methods

    `get_num_tokens(self, text: str) ‑> int`
    :   Get the number of tokens given the text string.
        
        Args:
            text (str): Text
        
        Returns:
            int: Number of tokens

    `get_token_ids(self, text: str) ‑> List[int]`
    :   Get the token ids of the given text.
        
        Args:
            text (str): Text
        
        Returns:
            List[int]: List of token ids.

    `stream(self, input: str, config: Optional[langchain_core.runnables.config.RunnableConfig] = None, *, stop: Optional[List[str]] = None, **kwargs) ‑> Iterator[str]`
    :   Text streaming of llm generation. Return a python generator of output tokens of the llm given the prompt.
        
        Args:
            input (str): The prompt to the llm.
            config (Optional[RunnableConfig]): Not used. Defaults to None.
            stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. If provided, it will overide the original llm stop list. Defaults to None.
        
        Yields:
            Iterator[str]: The next generated token.

`DebugLLM(core: llmplus.Models.Cores.base_core.BaseCore, temperature: float = 0, max_new_tokens: int = 2048, top_p: float = 0.95, top_k: int = 40, repetition_penalty: float = 1.1, stop: Optional[List[str]] = None, stop_newline_version: bool = True)`
:   Base LLM class for llmplus, using the LLM class from langchain.
        
    
    Initialising the LLM.
    
    Args:
        core (BaseCore): The BaseCore core.
        temperature (float, optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to 0.
        max_new_tokens (int, optional): Maximum number of tokens to generate by the llm. Defaults to 2048.
        top_p (float, optional): While sampling the next token, only consider the tokens above this p value. Defaults to 0.95.
        top_k (int, optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to 40.
        repetition_penalty (float, optional): The value to penalise the model for generating repetitive text. Defaults to 1.1.
        stop (Optional[List[str]], optional): List of strings to stop the generation of the llm. Defaults to None.
        stop_newline_version (bool, optional): Whether to add duplicates of the list of stop words starting with a new line character. Defaults to True.

    ### Ancestors (in MRO)

    * llmplus.Models.Cores.base_core.BaseLLM
    * langchain_core.language_models.llms.LLM
    * langchain_core.language_models.llms.BaseLLM
    * langchain_core.language_models.base.BaseLanguageModel
    * langchain_core.runnables.base.RunnableSerializable
    * langchain_core.load.serializable.Serializable
    * pydantic.v1.main.BaseModel
    * pydantic.v1.utils.Representation
    * langchain_core.runnables.base.Runnable
    * typing.Generic
    * abc.ABC

    ### Class variables

    `core: llmplus.Models.Cores.base_core.BaseCore`
    :

    `generation_config: Dict[str, Any]`
    :

    `stop: List[str]`
    :