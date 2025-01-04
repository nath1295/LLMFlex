from .base_engine import BaseEngine
import os
from typing import Optional, Any, List, Dict, Iterator, Literal, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from openai import Client

KNOWN_BACKEND = Literal['openai', 'mlx-textgen', 'vllm', 'llama.cpp', 'llama-cpp-python']

class OpenAIEngine(BaseEngine):
    """Class for LLM engine using openai API.
    """
    def __init__(self,
            model_id: Optional[str],
            tokenizer_name_or_path: Optional[str], 
            tokenizer_kwargs: Optional[Dict[str, Any]] = None, 
            base_url: Optional[str] = None,
            api_engine: Optional[KNOWN_BACKEND] = None,
            api_key: Optional[str] = None,
            **kwargs
        ) -> None:
        """Initializes the OpenAIEngine with the given parameters.

        Args:
            model_id (str): The ID of the OpenAI model to use. If None is provided, the first model in the list of models from the API backend will be used.
            tokenizer_name_or_path (Optional[str]): The name or path of the tokenizer to use. If None is given, it will use the tiktoken tokenizer from openai.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            base_url (Optional[str], optional): The base URL for the OpenAI API. Defaults to None.
            api_engine (Optional[KNOWN_BACKEND], optional): The backend LLM engine. This will help to decide the way of doing structured text generation. If not provided, not all structured text generation methods might work properly. Defaults to None.
            api_key (Optional[str], optional): The API key for the OpenAI API. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the BaseEngine initializer.
        """
        from openai import Client
        tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        base_url = os.environ.get('OPENAI_BASE_URL') if base_url is None else base_url
        api_key = os.environ.get('OPENAI_API_KEY') if api_key is None else api_key

        self._text_key = kwargs.pop('text_key', 'text') # key to get the generated text in the choices of the generated object. With llama.cpp server, it is 'content'.
        default_chat_template = kwargs.pop('default_chat_template', None)
        model = Client(base_url=base_url, api_key=api_key, **kwargs)
        avail_models = [m.id for m in model.models.list().data]
        model_id = avail_models[0] if model_id is None else model_id
        if model_id not in avail_models:
            raise ModuleNotFoundError(f'Model "{model_id}" does not exist.')
        self._api_engine = api_engine
        
        if (api_engine == 'mlx-textgen') and (tokenizer_name_or_path is None):
            tokenizer_name_or_path = model.models.retrieve(model_id).info['tokenizer_id']

        if tokenizer_name_or_path:
            from ...Tokenizer.huggingface_tokenizer import HuggingFaceTokenizer
            tokenizer = HuggingFaceTokenizer(pretrained_model_name_or_path=tokenizer_name_or_path, **tokenizer_kwargs)
        else:
            from ...Tokenizer.openai_tokenizer import OpenAITokenizer
            tokenizer = OpenAITokenizer(model_id=model_id)
        model_name = kwargs.get('model_name', None)
        if tokenizer.__class__.__name__ == 'OpenAITokenizer':
            import warnings
            warnings.warn('"tokenizer_name_or_path" not provided, the engine is assuming the given "base_url" is an official OpenAI url.')
            self._api_engine = 'openai'
        if self._api_engine == 'llama.cpp':
            self._text_key = 'content'
        super().__init__(model, model_id, tokenizer, model_name, default_chat_template)
    
    @property
    def model(self) -> "Client":
        """LLM model.

        Returns:
            Any: LLM model.
        """
        return self._model

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
        extra_body = kwargs.get('extra_body', dict())
        if top_k:
            extra_body['top_k'] = top_k
        if min_p:
            extra_body['min_p'] = min_p
        kwargs.pop('stream', None)
        gen_kwargs = dict(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=kwargs.pop('frequency_penalty', repetition_penalty),
            seed=seed,
            stop=stop,
            stream=False,
            extra_body=extra_body
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        gen_kwargs.update(kwargs)

        outputs = self.model.completions.create(
            model=self.model_id,
            prompt=prompts,
            **gen_kwargs
        ).choices
        outputs = [o.model_dump()[self._text_key] for o in outputs]
        return outputs


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
        extra_body = kwargs.get('extra_body', dict())
        if top_k:
            extra_body['top_k'] = top_k
        if min_p:
            extra_body['min_p'] = min_p
        kwargs.pop('stream', None)
        gen_kwargs = dict(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=kwargs.pop('frequency_penalty', repetition_penalty),
            seed=seed,
            stop=stop,
            stream=True,
            extra_body=extra_body
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        gen_kwargs.update(kwargs)

        outputs = self.model.completions.create(
            model=self.model_id,
            prompt=prompts,
            **gen_kwargs
        )
        num_prompt = len(prompts)
        def _gen_tokens():
            for i in outputs:
                outs = [''] * num_prompt
                choices = i.choices
                for o in choices:
                    odict = o.model_dump()
                    index = odict['index']
                    outs[index] = odict[self._text_key]
                yield outs
        return _gen_tokens()
    
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
        arg_dict = dict(
            prompts=prompts,
            stop=[self.tokenizer.eos_token],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed
        )
        arg_dict.update(kwargs)
        if self._api_engine in ['mlx-textgen', 'vllm']:
            extra_body = arg_dict.get('extra_body', dict())
            extra_body['guided_json'] = json_schema
            arg_dict['extra_body'] = extra_body
        else:
            api_engine = self._api_engine if self._api_engine is not None else 'unknown'
            raise NotImplementedError(f'"batch_generate_json" not implemented for "{self.__class__.__name__}" with {api_engine} backend.')
        
        if stream:
            return self.batch_generate_stream(**arg_dict)
        else:
            return self.batch_generate(**arg_dict)

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
        arg_dict = dict(
            prompts=prompts,
            stop=[self.tokenizer.eos_token],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed
        )
        arg_dict.update(kwargs)
        if self._api_engine in ['mlx-textgen', 'vllm']:
            extra_body = arg_dict.get('extra_body', dict())
            extra_body['guided_choice'] = choices
            arg_dict['extra_body'] = extra_body
        else:
            api_engine = self._api_engine if self._api_engine is not None else 'unknown'
            raise NotImplementedError(f'"batch_generate_choice" not implemented for "{self.__class__.__name__}" with {api_engine} backend.')
        
        if stream:
            return self.batch_generate_stream(**arg_dict)
        else:
            return self.batch_generate(**arg_dict)

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
        arg_dict = dict(
            prompts=prompts,
            stop=[self.tokenizer.eos_token],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed
        )
        arg_dict.update(kwargs)
        if self._api_engine in ['mlx-textgen', 'vllm']:
            extra_body = arg_dict.get('extra_body', dict())
            extra_body['guided_regex'] = regex
            arg_dict['extra_body'] = extra_body
        else:
            api_engine = self._api_engine if self._api_engine is not None else 'unknown'
            raise NotImplementedError(f'"batch_generate_regex" not implemented for "{self.__class__.__name__}" with {api_engine} backend.')
        
        if stream:
            return self.batch_generate_stream(**arg_dict)
        else:
            return self.batch_generate(**arg_dict)

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
        arg_dict = dict(
            prompts=prompts,
            stop=[self.tokenizer.eos_token],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed
        )
        arg_dict.update(kwargs)
        if self._api_engine in ['mlx-textgen', 'vllm']:
            extra_body = arg_dict.get('extra_body', dict())
            extra_body['guided_grammar'] = cfg_grammar
            arg_dict['extra_body'] = extra_body
        else:
            api_engine = self._api_engine if self._api_engine is not None else 'unknown'
            raise NotImplementedError(f'"batch_generate_grammar" not implemented for "{self.__class__.__name__}" with {api_engine} backend.')
        
        if stream:
            return self.batch_generate_stream(**arg_dict)
        else:
            return self.batch_generate(**arg_dict)