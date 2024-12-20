from ...Tokenizer.base_tokenizer import BaseTokenizer
from ...Prompt.chat_template import ChatTemplate, CHAT_TEMPLATE_PRESETS
from typing import List, Optional, Any, Iterator, Dict, Union
from abc import ABC, abstractmethod

class BaseEngine(ABC):
    """Base class for LLM engine.
    """
    def __init__(self, model: Any, model_id: str, tokenizer: BaseTokenizer, model_name: Optional[str] = None, default_chat_template: Optional[CHAT_TEMPLATE_PRESETS] = None) -> None:
        self._model = model
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._default_chat_template = default_chat_template

    @property
    def model(self) -> Any:
        """LLM model.

        Returns:
            Any: LLM model.
        """
        return self._model
    
    @property
    def model_id(self) -> str:
        """Model ID.

        Returns:
            str: Model ID.
        """
        return self._model_id
    
    @property
    def model_name(self) -> str:
        """Prettified version of model ID.

        Returns:
            str: Prettified version of model ID.
        """
        if self._model_name is None:
            import re
            model_name = re.split('\\\|/', self.model_id)[-1]
            self._model_name = model_name.replace('_', '-').lower().strip('-').strip()
        return self._model_name
    
    @property
    def tokenizer(self) -> BaseTokenizer:
        """Tokenizer.

        Returns:
            BaseTokenizer: Tokenizer.
        """
        return self._tokenizer
    
    @property
    def chat_template(self) -> ChatTemplate:
        """Chat template for the LLM engine.

        This property returns a ChatTemplate object that can be used to format input prompts for the LLM engine.
        
        Returns:
            ChatTemplate: Chat template for the LLM engine.
        """
        if not hasattr(self, '_chat_template'):
            from ...Prompt.presets import PRESETS
            chat_template = self._default_chat_template
            chat_template = None if chat_template not in PRESETS.keys() else chat_template
            self._chat_template = ChatTemplate(tokenizer=self.tokenizer, chat_template=chat_template)
        return self._chat_template
    
    @abstractmethod
    def batch_generate(self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
        ) -> List[str]:
        """Generates text for a batch of prompts using the LLM model.

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
        """
        pass

    @abstractmethod
    def batch_generate_stream(self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
        ) -> Iterator[List[str]]:
        """Generates text for a batch of prompts using the LLM model in a streaming manner.

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
        """
        pass

    def batch_generate_json(self,
            prompts: List[str],
            json_schema: Dict[str, Any],
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stream: bool = False,
            **kwargs
        ) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a batch of prompts using the LLM model, following a given JSON schema.

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
        """
        raise NotImplementedError(f'"batch_generate_json" not implemented for "{self.__class__.__name__}".')

    def batch_generate_choice(self,
            prompts: List[str],
            choices: List[str],
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stream: bool = False,
            **kwargs
        ) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a batch of prompts using the LLM model, choosing from a list of options.

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
        """
        raise NotImplementedError(f'"batch_generate_choice" not implemented for "{self.__class__.__name__}".')

    def batch_generate_regex(self,
            prompts: List[str],
            regex: str,
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stream: bool = False,
            **kwargs
        ) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a batch of prompts using the LLM model, ensuring it matches a given regular expression.

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
        """
        raise NotImplementedError(f'"batch_generate_regex" not implemented for "{self.__class__.__name__}".')

    def batch_generate_grammar(self,
            prompts: List[str],
            cfg_grammar: str,
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stream: bool = False,
            **kwargs
        ) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a batch of prompts using the LLM model, ensuring it adheres to a given grammar.

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
        """
        raise NotImplementedError(f'"batch_generate_grammar" not implemented for "{self.__class__.__name__}".')

    def generate(self, 
            prompt: str,
            stop: Optional[List[str]] = None,
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
        ) -> str:
        """Generates text for a single prompt using the LLM model.

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
        """
        return self.batch_generate(
            prompts=[prompt],
            stop=stop,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            **kwargs
        )[0]

    def generate_stream(self, 
            prompt: str,
            stop: Optional[List[str]] = None,
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
        ) -> Iterator[str]:
        """Generates text for a single prompt using the LLM model in a streaming manner.

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
        """
        def generate():
            for tokens in self.batch_generate_stream(
                prompts=[prompt],
                stop=stop,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed,
                **kwargs
            ):
                yield tokens[0]
        return generate()

    def unload(self) -> None:
        """Unload the LLM model from memory.
        """
        import gc
        del self._model, self._tokenizer
        gc.collect()

class BaseLLM:
    """LLM class for text generation. 
    """
    def __init__(self,
            engine: BaseEngine,
            temperature: float = 0.0,
            max_new_tokens: int = 512,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            top_k: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            stop: Optional[List[str]] = None,
            seed: Optional[int] = None,
            **kwargs        
        ) -> None:
        """Initialise the LLM.

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
        """
        self._engine = engine
        stop = stop if stop else [self.tokenizer.eos_token]
        stop = list(set(stop))
        if self.tokenizer.eos_token not in stop:
            stop.append(self.tokenizer.eos_token)
        self._gen_kwargs = dict(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
            seed=seed
        )
        self._gen_kwargs.update(kwargs)

    @property
    def engine(self) -> BaseEngine:
        """Engine used in the LLM.

        Returns:
            BaseEngine: Engine used in the LLM.
        """
        return self._engine

    @property
    def tokenizer(self) -> BaseTokenizer:
        """Returns the tokenizer used in the LLM.

        Returns:
            BaseTokenizer: The tokenizer used in the LLM.
        """
        return self.engine.tokenizer
    
    @property
    def chat_template(self) -> ChatTemplate:
        """Returns the default chat template for the LLM.

        Returns:
            ChatTemplate: The default chat template for the LLM.
        """
        return self.engine.chat_template
    
    @property
    def model_id(self) -> str:
        """Returns the ID of the LLM model.

        Returns:
            str: The ID of the LLM model.
        """
        return self.engine.model_id
    
    @property
    def model_name(self) -> str:
        """Returns the name of the LLM model.

        Returns:
            str: The name of the LLM model.
        """
        return self.engine.model_name
    
    @property
    def gen_kwargs(self) -> Dict[str, Any]:
        """Returns the generation parameters used in the LLM.

        These parameters are used in the `generate` and `stream` methods.
        They include temperature, max_new_tokens, top_p, min_p, top_k, repetition_penalty,
        stop, and seed.

        Returns:
            Dict[str, Any]: A dictionary containing the generation parameters.
        """
        return self._gen_kwargs
    
    def batch_generate(self, prompts: List[str], stream: bool = False, **kwargs) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a list of prompts using the LLM model.

        This method takes a list of prompts and generates text for each prompt using the LLM model.

        Args:
            prompts (List[str]): A list of input prompts.
            stream (bool, optional): Whether to stream the generated text. Defaults to False.
            **kwargs: Additional generation parameters to overwrite the default parameters.

        Returns:
            Union[List[str], Iterator[List[str]]]: If `stream` is False, returns a list of generated texts.
            If `stream` is True, returns a generator that yields a list of generated texts, one token at a time.
        """
        from copy import deepcopy
        gen_kwargs = deepcopy(self._gen_kwargs)
        gen_kwargs.update(kwargs)
        if stream:
            return self.engine.batch_generate_stream(prompts=prompts, **gen_kwargs)
        else:
            return self.engine.batch_generate(prompts=prompts, **gen_kwargs)
        
    def batch_generate_json(
            self,
            prompts: List[str],
            json_schema: Dict[str, Any],
            stream: bool = False,
            **kwargs
        ) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a list of prompts using the LLM model and a JSON schema.

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
        """
        from copy import deepcopy
        gen_kwargs = dict(
            prompts=prompts,
            json_schema=json_schema,
            stream=stream
        )
        gen_kwargs.update(deepcopy(self._gen_kwargs))
        gen_kwargs.update(kwargs)
        gen_kwargs.pop('stop', None)
        return self.engine.batch_generate_json(**gen_kwargs)
    
    def batch_generate_choice(
            self,
            prompts: List[str],
            choices: List[str],
            stream: bool = False,
            **kwargs
        ) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a list of prompts using the LLM model and a list of choices.

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
        """
        from copy import deepcopy
        gen_kwargs = dict(
            prompts=prompts,
            choices=choices,
            stream=stream
        )
        gen_kwargs.update(deepcopy(self._gen_kwargs))
        gen_kwargs.update(kwargs)
        gen_kwargs.pop('stop', None)
        return self.engine.batch_generate_choice(**gen_kwargs)
    
    def batch_generate_regex(
            self,
            prompts: List[str],
            regex: str,
            stream: bool = False,
            **kwargs
        ) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a list of prompts using the LLM model and a regular expression.

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
        """
        from copy import deepcopy
        gen_kwargs = dict(
            prompts=prompts,
            regex=regex,
            stream=stream
        )
        gen_kwargs.update(deepcopy(self._gen_kwargs))
        gen_kwargs.update(kwargs)
        gen_kwargs.pop('stop', None)
        return self.engine.batch_generate_regex(**gen_kwargs)
    
    def batch_generate_grammar(
            self,
            prompts: List[str],
            cfg_grammar: str,
            stream: bool = False,
            **kwargs
        ) -> Union[List[str], Iterator[List[str]]]:
        """Generates text for a list of prompts using the LLM model and a grammar configuration.

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
        """
        from copy import deepcopy
        gen_kwargs = dict(
            prompts=prompts,
            cfg_grammar=cfg_grammar,
            stream=stream
        )
        gen_kwargs.update(deepcopy(self._gen_kwargs))
        gen_kwargs.update(kwargs)
        gen_kwargs.pop('stop', None)
        return self.engine.batch_generate_grammar(**gen_kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generates text for a single prompt using the LLM model.

        This method takes a single prompt and generates text using the LLM model.
        Args:
            prompt (str): The input prompt.
            **kwargs: Additional generation parameters to overwrite the default parameters.

        Returns:
            str: The generated text.
        """
        return self.batch_generate(prompts=[prompt], stream=False, **kwargs)[0]
    
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generates text for a single prompt using the LLM model in a streaming manner.

        This method takes a single prompt and generates text using the LLM model.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional generation parameters to overwrite the default parameters.

        Yields:
            Iterator[str]: A generator that yields the generated text, one token at a time.
        """
        from copy import deepcopy
        gen_kwargs = deepcopy(self._gen_kwargs)
        gen_kwargs.update(kwargs)
        return self.engine.generate_stream(prompt=prompt, **gen_kwargs)
    
    def generate_json(self, prompt: str, json_schema: Dict[str, Any], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Generates text for a single prompt using the LLM model and a JSON schema.

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
        """
        output = self.batch_generate_json(prompts=[prompt], json_schema=json_schema, stream=stream, **kwargs)
        if stream:
            def output_iterator():
                for tokens in output:
                    yield tokens[0]
            return output_iterator()
        else:
            return output[0]

    def generate_choice(self, prompt: str, choices: List[str], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Generates text for a single prompt using the LLM model and a list of choices.

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
        """
        output = self.batch_generate_choice(prompts=[prompt], choices=choices, stream=stream, **kwargs)
        if stream:
            def output_iterator():
                for tokens in output:
                    yield tokens[0]
            return output_iterator()
        else:
            return output[0]
        
    def generate_regex(self, prompt: str, regex: str, stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Generates text for a single prompt using the LLM model and a regular expression.

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
        """
        output = self.batch_generate_regex(prompts=[prompt], regex=regex, stream=stream, **kwargs)
        if stream:
            def output_iterator():
                for tokens in output:
                    yield tokens[0]
            return output_iterator()
        else:
            return output[0]
    
    def generate_grammar(self, prompt: str, cfg_grammar: str, stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Generates text for a single prompt using the LLM model and a grammar configuration.

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
        """
        output = self.batch_generate_regex(prompts=[prompt], cfg_grammar=cfg_grammar, stream=stream, **kwargs)
        if stream:
            def output_iterator():
                for tokens in output:
                    yield tokens[0]
            return output_iterator()
        else:
            return output[0]
