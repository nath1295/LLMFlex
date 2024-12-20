Module llmflex.LLM.Engine.base_engine
=====================================

Classes
-------

`BaseEngine(model: Any, model_id: str, tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer, model_name: str | None = None, default_chat_template: Literal['chatml', 'llama3', 'mistral', 'gemma', 'deepseek', 'openchat', 'phi'] | None = None)`
:   Base class for LLM engine.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.LLM.Engine.huggingface_engine.HuggingFaceEngine
    * llmflex.LLM.Engine.llamacpp_engine.LlamaCppEngine
    * llmflex.LLM.Engine.mlx_engine.MLXEngine
    * llmflex.LLM.Engine.openai_engine.OpenAIEngine

    ### Instance variables

    `chat_template: llmflex.Prompt.chat_template.ChatTemplate`
    :   Chat template for the LLM engine.
        
        This property returns a ChatTemplate object that can be used to format input prompts for the LLM engine.
        
        Returns:
            ChatTemplate: Chat template for the LLM engine.

    `model: Any`
    :   LLM model.
        
        Returns:
            Any: LLM model.

    `model_id: str`
    :   Model ID.
        
        Returns:
            str: Model ID.

    `model_name: str`
    :   Prettified version of model ID.
        
        Returns:
            str: Prettified version of model ID.

    `tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer`
    :   Tokenizer.
        
        Returns:
            BaseTokenizer: Tokenizer.

    ### Methods

    `batch_generate(self, prompts: List[str], stop: List[str] | None = None, temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, seed: int | None = None, **kwargs) ‑> List[str]`
    :   Generates text for a batch of prompts using the LLM model.
        
        This method takes a list of prompts and generates text for each prompt using the LLM model.
        It supports various generation parameters such as temperature, max_new_tokens, top_p, min_p,
        top_k, repetition_penalty, and seed.
        
        Args:
            prompts (List[str]): A list of input prompts.
            stop (Optional[List[str]], optional): A list of stop words to end generation. Defaults to None.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
        
        Returns:
            List[str]: A list of generated texts, one for each input prompt.

    `batch_generate_choice(self, prompts: List[str], choices: List[str], temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, seed: int | None = None, stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a batch of prompts using the LLM model, choosing from a list of options.
        
        This method takes a list of prompts and generates text for each prompt using the LLM model.
        It supports various generation parameters such as temperature, max_new_tokens, top_p, min_p,
        top_k, repetition_penalty, and seed. The generated text is chosen from a list of options.
        
        Args:
            prompts (List[str]): A list of input prompts.
            choices (List[str]): A list of options to choose from.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
            stream (bool, optional): Whether to stream the generated text one token at a time. Defaults to False.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: A list of generated texts, one for each input prompt,
                chosen from the list of options. If `stream` is True, a generator that yields the generated
                texts one token at a time is returned instead.

    `batch_generate_grammar(self, prompts: List[str], cfg_grammar: str, temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, seed: int | None = None, stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a batch of prompts using the LLM model, ensuring it adheres to a given grammar.
        
        This method takes a list of prompts and generates text for each prompt using the LLM model.
        It supports various generation parameters such as temperature, max_new_tokens, top_p, min_p,
        top_k, repetition_penalty, and seed. The generated text is ensured to adhere to the provided grammar.
        
        Args:
            prompts (List[str]): A list of input prompts.
            cfg_grammar (str): A string representing the grammar to adhere to.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
            stream (bool, optional): Whether to stream the generated text one token at a time. Defaults to False.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: A list of generated texts, one for each input prompt,
                that adhere to the provided grammar. If `stream` is True, a generator that yields the generated
                texts one token at a time is returned instead.

    `batch_generate_json(self, prompts: List[str], json_schema: Dict[str, Any], temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, seed: int | None = None, stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a batch of prompts using the LLM model, following a given JSON schema.
        
        This method takes a list of prompts and generates text for each prompt using the LLM model.
        It supports various generation parameters such as temperature, max_new_tokens, top_p, min_p,
        top_k, repetition_penalty, and seed. The generated text is formatted according to the provided JSON schema.
        
        Args:
            prompts (List[str]): A list of input prompts.
            json_schema (Dict[str, Any]): A JSON schema defining the structure of the generated text.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
            stream (bool, optional): Whether to stream the generated text one token at a time. Defaults to False.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: A list of generated texts, one for each input prompt,
                formatted according to the JSON schema. If `stream` is True, a generator that yields the generated
                texts one token at a time is returned instead.

    `batch_generate_regex(self, prompts: List[str], regex: str, temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, seed: int | None = None, stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a batch of prompts using the LLM model, ensuring it matches a given regular expression.
        
        This method takes a list of prompts and generates text for each prompt using the LLM model.
        It supports various generation parameters such as temperature, max_new_tokens, top_p, min_p,
        top_k, repetition_penalty, and seed. The generated text is ensured to match the provided regular expression.
        
        Args:
            prompts (List[str]): A list of input prompts.
            regex (str): A regular expression to match the generated text against.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
            stream (bool, optional): Whether to stream the generated text one token at a time. Defaults to False.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: A list of generated texts, one for each input prompt,
                that match the provided regular expression. If `stream` is True, a generator that yields the generated
                texts one token at a time is returned instead.

    `batch_generate_stream(self, prompts: List[str], stop: List[str] | None = None, temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, seed: int | None = None, **kwargs) ‑> Iterator[List[str]]`
    :   Generates text for a batch of prompts using the LLM model in a streaming manner.
        
        This method takes a list of prompts and generates text for each prompt using the LLM model.
        It supports various generation parameters such as temperature, max_new_tokens, top_p, min_p,
        top_k, repetition_penalty, and seed. The generated text is yielded one token at a time.
        Args:
            prompts (List[str]): A list of input prompts.
            stop (Optional[List[str]], optional): A list of stop words to end generation. Defaults to None.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
        
        Yields:
            Iterator[List[str]]: A generator that yields a list of generated texts, one for each input prompt,
                one token at a time.

    `generate(self, prompt: str, stop: List[str] | None = None, temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, seed: int | None = None, **kwargs) ‑> str`
    :   Generates text for a single prompt using the LLM model.
        
        This method takes a single prompt and generates text using the LLM model.
        It supports various generation parameters such as temperature, max_new_tokens, top_p, min_p,
        top_k, repetition_penalty, and seed.
        
        Args:
            prompt (str): The input prompt.
            stop (Optional[List[str]], optional): A list of stop words to end generation. Defaults to None.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
        
        Returns:
            str: The generated text.

    `generate_stream(self, prompt: str, stop: List[str] | None = None, temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, seed: int | None = None, **kwargs) ‑> Iterator[str]`
    :   Generates text for a single prompt using the LLM model in a streaming manner.
        
        This method takes a single prompt and generates text using the LLM model.
        It supports various generation parameters such as temperature, max_new_tokens, top_p, min_p,
        top_k, repetition_penalty, and seed. The generated text is yielded one token at a time.
        
        Args:
            prompt (str): The input prompt.
            stop (Optional[List[str]], optional): A list of stop words to end generation. Defaults to None.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
            top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
            min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
            top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
            repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
            seed (Optional[int], optional): The seed for generation. Defaults to None.
        
        Yields:
            Iterator[str]: A generator that yields the generated text, one token at a time.

    `unload(self) ‑> None`
    :   Unload the LLM model from memory.

`BaseLLM(engine: llmflex.LLM.Engine.base_engine.BaseEngine, temperature: float = 0.0, max_new_tokens: int = 512, top_p: float | None = None, min_p: float | None = None, top_k: int | None = None, repetition_penalty: float | None = None, stop: List[str] | None = None, seed: int | None = None, **kwargs)`
:   LLM class for text generation. 
        
    
    Initialise the LLM.
    
    Args:
        engine (BaseEngine): Engine used in the LLM.
        temperature (float, optional): The temperature to use for generation. Defaults to 0.0.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
        top_p (Optional[float], optional): The top-p parameter for generation. Defaults to None.
        min_p (Optional[float], optional): The min-p parameter for generation. Defaults to None.
        top_k (Optional[int], optional): The top-k parameter for generation. Defaults to None.
        repetition_penalty (Optional[float], optional): The repetition penalty for generation. Defaults to None.
        stop (Optional[List[str]], optional): A list of stop words to end generation. Defaults to None.
        seed (Optional[int], optional): The seed for generation. Defaults to None.

    ### Instance variables

    `chat_template: llmflex.Prompt.chat_template.ChatTemplate`
    :   Returns the default chat template for the LLM.
        
        Returns:
            ChatTemplate: The default chat template for the LLM.

    `engine: llmflex.LLM.Engine.base_engine.BaseEngine`
    :   Engine used in the LLM.
        
        Returns:
            BaseEngine: Engine used in the LLM.

    `gen_kwargs: Dict[str, Any]`
    :   Returns the generation parameters used in the LLM.
        
        These parameters are used in the `generate` and `stream` methods.
        They include temperature, max_new_tokens, top_p, min_p, top_k, repetition_penalty,
        stop, and seed.
        
        Returns:
            Dict[str, Any]: A dictionary containing the generation parameters.

    `model_id: str`
    :   Returns the ID of the LLM model.
        
        Returns:
            str: The ID of the LLM model.

    `model_name: str`
    :   Returns the name of the LLM model.
        
        Returns:
            str: The name of the LLM model.

    `tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer`
    :   Returns the tokenizer used in the LLM.
        
        Returns:
            BaseTokenizer: The tokenizer used in the LLM.

    ### Methods

    `batch_generate(self, prompts: List[str], stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a list of prompts using the LLM model.
        
        This method takes a list of prompts and generates text for each prompt using the LLM model.
        
        Args:
            prompts (List[str]): A list of input prompts.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: If `stream` is False, returns a list of generated texts.
            If `stream` is True, returns a generator that yields a list of generated texts, one token at a time.

    `batch_generate_choice(self, prompts: List[str], choices: List[str], stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a list of prompts using the LLM model and a list of choices.
        
        This method takes a list of prompts and a list of choices, and generates text for each prompt
        using the LLM model. The generated text is expected to be one of the choices provided.
        
        Args:
            prompts (List[str]): A list of input prompts.
            choices (List[str]): A list of possible choices for the generated text.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: If `stream` is False, returns a list of generated texts.
            If `stream` is True, returns a generator that yields a list of generated texts, one token at a time.

    `batch_generate_grammar(self, prompts: List[str], cfg_grammar: str, stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a list of prompts using the LLM model and a grammar configuration.
        
        This method takes a list of prompts and a grammar configuration, and generates text for each prompt
        using the LLM model. The generated text is expected to follow the grammar rules defined by the
        configuration.
        
        Args:
            prompts (List[str]): A list of input prompts.
            cfg_grammar (str): A grammar configuration defining the grammar rules for the generated text.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: If `stream` is False, returns a list of generated texts.
            If `stream` is True, returns a generator that yields a list of generated texts, one token at a time.

    `batch_generate_json(self, prompts: List[str], json_schema: Dict[str, Any], stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a list of prompts using the LLM model and a JSON schema.
        
        This method takes a list of prompts and a JSON schema, and generates text for each prompt
        using the LLM model. The generated text is expected to follow the structure defined by the
        JSON schema.
        
        Args:
            prompts (List[str]): A list of input prompts.
            json_schema (Dict[str, Any]): A JSON schema defining the structure of the generated text.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: If `stream` is False, returns a list of generated texts.
            If `stream` is True, returns a generator that yields a list of generated texts, one token at a time.

    `batch_generate_regex(self, prompts: List[str], regex: str, stream: bool = False, **kwargs) ‑> List[str] | Iterator[List[str]]`
    :   Generates text for a list of prompts using the LLM model and a regular expression.
        
        This method takes a list of prompts and a regular expression, and generates text for each prompt
        using the LLM model. The generated text is expected to match the regular expression provided.
        
        Args:
            prompts (List[str]): A list of input prompts.
            regex (str): A regular expression defining the structure of the generated text.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[List[str], Iterator[List[str]]]: If `stream` is False, returns a list of generated texts.
            If `stream` is True, returns a generator that yields a list of generated texts, one token at a time.

    `generate(self, prompt: str, **kwargs) ‑> str`
    :   Generates text for a single prompt using the LLM model.
        
        This method takes a single prompt and generates text using the LLM model.
        Args:
            prompt (str): The input prompt.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            str: The generated text.

    `generate_choice(self, prompt: str, choices: List[str], stream: bool = False, **kwargs) ‑> str | Iterator[str]`
    :   Generates text for a single prompt using the LLM model and a list of choices.
        
        This method takes a single prompt and a list of choices, and generates text using the LLM model.
        The generated text is expected to be one of the choices provided.
        
        Args:
            prompt (str): The input prompt.
            choices (List[str]): A list of possible choices for the generated text.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[str, Iterator[str]]: If `stream` is False, returns the generated text.
            If `stream` is True, returns a generator that yields the generated text, one token at a time.

    `generate_grammar(self, prompt: str, cfg_grammar: str, stream: bool = False, **kwargs) ‑> str | Iterator[str]`
    :   Generates text for a single prompt using the LLM model and a grammar configuration.
        
        This method takes a single prompt and a grammar configuration, and generates text using the LLM model.
        The generated text is expected to follow the grammar rules defined by the configuration.
        
        Args:
            prompt (str): The input prompt.
            cfg_grammar (str): A grammar configuration defining the grammar rules for the generated text.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[str, Iterator[str]]: If `stream` is False, returns the generated text.
            If `stream` is True, returns a generator that yields the generated text, one token at a time.

    `generate_json(self, prompt: str, json_schema: Dict[str, Any], stream: bool = False, **kwargs) ‑> str | Iterator[str]`
    :   Generates text for a single prompt using the LLM model and a JSON schema.
        
        This method takes a single prompt and a JSON schema, and generates text using the LLM model.
        The generated text is expected to follow the structure defined by the JSON schema.
        
        Args:
            prompt (str): The input prompt.
            json_schema (Dict[str, Any]): A JSON schema defining the structure of the generated text.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[str, Iterator[str]]: If `stream` is False, returns the generated text.
            If `stream` is True, returns a generator that yields the generated text, one token at a time.

    `generate_regex(self, prompt: str, regex: str, stream: bool = False, **kwargs) ‑> str | Iterator[str]`
    :   Generates text for a single prompt using the LLM model and a regular expression.
        
        This method takes a single prompt and a regular expression, and generates text using the LLM model.
        The generated text is expected to match the regular expression provided.
        
        Args:
            prompt (str): The input prompt.
            regex (str): A regular expression defining the structure of the generated text.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Returns:
            Union[str, Iterator[str]]: If `stream` is False, returns the generated text.
            If `stream` is True, returns a generator that yields the generated text, one token at a time.

    `stream(self, prompt: str, **kwargs) ‑> Iterator[str]`
    :   Generates text for a single prompt using the LLM model in a streaming manner.
        
        This method takes a single prompt and generates text using the LLM model.
        
        Args:
            prompt (str): The input prompt.
            **kwargs: Additional generation parameters to overwrite the default parameters.
        
        Yields:
            Iterator[str]: A generator that yields the generated text, one token at a time.