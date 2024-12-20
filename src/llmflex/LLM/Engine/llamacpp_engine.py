from .base_engine import BaseEngine
import os
from typing import Optional, List, Iterator, Dict, Any, Union
try:
    from llama_cpp import Llama
    lcpp_installed = True
except:
    lcpp_installed = False

class LlamaCppEngine(BaseEngine):
    """Class for LLM engine using llama-cpp-python.
    """
    def __init__(self,
            pretrained_model_name_or_path: str, 
            model_file: Optional[str] = None,
            tokenizer_name_or_path: Optional[str] = None,
            context_size: int = 8192,
            verbose: bool = False,
            download_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> None:
        """Initializes the LlamaCppEngine with the given parameters.

        Args:
            pretrained_model_name_or_path (str): The HuggingFace repository name or the full path of the model file.
            model_file (str, optional): The model filename if a HuggingFace repository name is given for tokenizer_id_or_path. Defaults to None.
            tokenizer_name_or_path (str, optional): The Huggingface repository name or the full path to load the HuggingFace tokenizer. If not provided, structured text generation might encounter problems. Defaults to None.
            context_size (int, optional): The size of the context window. Defaults to 8192.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            download_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the download function. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the Llama class.
        """
        if not lcpp_installed:
            raise ModuleNotFoundError('"llama-cpp-python" not installed. To use LlamaCppEngine, please install "llama-cpp-python" by running "pip install llama-cpp-python".')
        from ...utils import get_config, download_file_from_repo
        from ...Tokenizer.llamacpp_tokenizer import LlamaCppTokenizer
        is_file = pretrained_model_name_or_path.endswith('.gguf')
        if model_file:
            file_dir = os.path.join(pretrained_model_name_or_path, model_file) if not is_file else pretrained_model_name_or_path
        else:
            file_dir = pretrained_model_name_or_path
        file_exist = os.path.exists(file_dir)
        load_kwargs = dict(
            verbose=verbose,
            n_ctx=context_size
        )
        kwargs.pop('verbose', None)
        kwargs.pop('n_ctx', None)
        default_chat_template = kwargs.pop('default_chat_template', None)
        load_kwargs.update(kwargs)
        if tokenizer_name_or_path:
            from llama_cpp.llama_tokenizer import LlamaHFTokenizer
            llama_tokenizer = LlamaHFTokenizer.from_pretrained(tokenizer_name_or_path)
            load_kwargs['tokenizer'] = llama_tokenizer
        else:
            import warnings
            warnings.warn('"tokenizer_name_or_path" is not provided. Might encounter issues while using structured text generation.')
        if file_exist:
            model = Llama(model_path=file_dir, **load_kwargs)
        else:
            download_kwargs = dict() if download_kwargs is None else download_kwargs
            cache_dir = download_kwargs.pop('cache_dir', get_config('hf_home'))
            file_dir = download_file_from_repo(repo_id=pretrained_model_name_or_path, filename=model_file, cache_dir=cache_dir, **download_kwargs)
            model = Llama(model_path=file_dir, **load_kwargs)
        tokenizer = LlamaCppTokenizer.from_llama_model(llama_model=model)
        model_id = os.path.basename(file_dir).removesuffix('.gguf')
        model_name = None

        super().__init__(model, model_id, tokenizer, model_name, default_chat_template)
    
    @property
    def model(self) -> Llama:
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
        outputs = []
        gen_kwargs = dict(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1 if top_p is None else top_p,
            min_p=0 if min_p is None else min_p,
            top_k=100 if top_k is None else top_k,
            stop=stop,
            repeat_penalty=1 if repetition_penalty is None else repetition_penalty,
            seed=seed,
            stream=False
        )
        for k in gen_kwargs.keys():
            kwargs.pop(k, None)
        gen_kwargs.update(kwargs)
        # Seems like llama-cpp-python does not support batch generation.
        for prompt in prompts:
            outputs.append(self.model.create_completion(prompt=prompt, **gen_kwargs)['choices'][0]['text'])
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
        gen_kwargs = dict(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1 if top_p is None else top_p,
            min_p=0 if min_p is None else min_p,
            top_k=100 if top_k is None else top_k,
            stop=stop,
            repeat_penalty=1 if repetition_penalty is None else repetition_penalty,
            seed=seed,
            stream=True
        )
        for k in gen_kwargs.keys():
            kwargs.pop(k, None)
        gen_kwargs.update(kwargs)
        num_prompt = len(prompts)

        def _gen_tokens():
            # This will result in streaming the prompts sequentially instead of asychronously, but there's no other way I can think of.
            for i, prompt in enumerate(prompts):
                for o in self.model.create_completion(prompt=prompt, **gen_kwargs):
                    outs = [''] * num_prompt
                    outs[i] = o['choices'][0]['text']
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
        from llama_cpp import LogitsProcessorList
        from .engine_utils import get_json_processor
        arg_dict = dict(
            prompts=prompts,
            stop=[self.tokenizer.eos_token],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            logits_processor=LogitsProcessorList([get_json_processor(schema=json_schema, tokenizer=self.tokenizer)])
        )
        arg_dict.update(kwargs)
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
        from llama_cpp import LogitsProcessorList
        from .engine_utils import get_choice_processor
        arg_dict = dict(
            prompts=prompts,
            stop=[self.tokenizer.eos_token],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            logits_processor=LogitsProcessorList([get_choice_processor(choices=choices, tokenizer=self.tokenizer)])
        )
        arg_dict.update(kwargs)
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
        from llama_cpp import LogitsProcessorList
        from .engine_utils import get_regex_processor
        arg_dict = dict(
            prompts=prompts,
            stop=[self.tokenizer.eos_token],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            logits_processor=LogitsProcessorList([get_regex_processor(regex_str=regex, tokenizer=self.tokenizer)])
        )
        arg_dict.update(kwargs)
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
        from llama_cpp import LogitsProcessorList
        from .engine_utils import get_grammar_processor
        arg_dict = dict(
            prompts=prompts,
            stop=[self.tokenizer.eos_token],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            logits_processor=LogitsProcessorList([get_grammar_processor(cfg_str=cfg_grammar, tokenizer=self.tokenizer)])
        )
        arg_dict.update(kwargs)
        if stream:
            return self.batch_generate_stream(**arg_dict)
        else:
            return self.batch_generate(**arg_dict)
    