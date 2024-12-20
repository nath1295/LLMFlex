from .base_engine import BaseEngine
import os
from typing import Optional, List, Iterator, Dict, Any, Union, NamedTuple, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.streamers import BaseStreamer

class TokenStreamer(BaseStreamer):
    """Class for streaming text tokens.
    """
    def __init__(self, pretrained_model_name_or_path: str, tokenizer: PreTrainedTokenizer) -> None:
        from queue import Queue
        from ...Tokenizer.hf_detokenizer import get_detokenizer
        self._detokenizer = get_detokenizer(pretrained_model_name_or_path, tokenizer)
        self.queue = Queue()
        self.start = False

    def put(self, value):
        if self.start:
            self._detokenizer.add_tokens(value[:, None].cpu().tolist())
            str_tokens = self._detokenizer.last_segments
            self.queue.put(str_tokens)
        else:
            # To eliminate the prompt.
            self.start = True
    
    def end(self):
        if self.start:
            self._detokenizer.finalize()
            str_tokens = self._detokenizer.last_segments
            self.queue.put(str_tokens)
        del self._detokenizer
        self.queue.put('ended')

    def __iter__(self):
        return self
    
    def __next__(self):
        val = self.queue.get()
        if val == 'ended':
            raise StopIteration()
        else:
            return val

class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int
    stop_text: Optional[str] = None

def stopping_criteria(
        text: str,
        stop_tuple: List[Tuple[str, int]]
    ) -> StopCondition:
    """Get the stopping condition with stop words and eos token.

    Args:
        text (str): Text to be determined to stop or not.
        stop_tuple (List[Tuple[str, int]]): List of tuple with the stop word string and the length of the string. Must be ordered descendingly by length.

    Returns:
        StopCondition: A status class stating whether the generation should stop and the length of text to trim.
    """
    return next(
        (StopCondition(stop_met=True, trim_length=len(text) - len(text.split(stop)[0]), stop_text=stop)
        for stop, length in stop_tuple if stop in text), StopCondition(stop_met=False, trim_length=0, stop_text=None)
    )

def sequence_overlap(s1: str, s2: str) -> bool:
    """Determine if there is overlapping string between the suffix of the first string and the prefix of the second string.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        bool: Whether there is overlap.
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))

class HuggingFaceEngine(BaseEngine):
    """Class for LLM engine using llama-cpp-python.
    """
    def __init__(self,
            pretrained_model_name_or_path: str,
            tokenizer_name_or_path: Optional[str] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> None:
        """Initializes the HuggingFaceEngine with the given parameters.

        Args:
            pretrained_model_name_or_path (str): The HuggingFace repository name or the full path of the model file.
            tokenizer_name_or_path (str, optional): The Huggingface repository name or the full path to load the HuggingFace tokenizer. If not provided, pretrained_model_name_or_path would be used. Defaults to None.
            model_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the model. Defaults to None.
            tokenizer_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the BaseEngine class.
        """
        try:
            import torch
        except:
            raise ModuleNotFoundError('"torch" not installed, Please install it with `pip install torch`.')
        from transformers import AutoModelForCausalLM
        from ...Tokenizer.huggingface_tokenizer import HuggingFaceTokenizer
        from ...utils import get_config
        model_kwargs = dict() if model_kwargs is None else model_kwargs
        tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        model_cache_dir = model_kwargs.pop('cache_dir', get_config('hf_home'))
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, cache_dir=model_cache_dir, **model_kwargs)
        tokenizer = HuggingFaceTokenizer(pretrained_model_name_or_path if tokenizer_name_or_path is None else tokenizer_name_or_path, **tokenizer_kwargs)
        # handling pad tokens
        if not tokenizer.pad_token:
            tokenizer._pad_token = tokenizer.eos_token
            tokenizer._pad_token_id = tokenizer.eos_token_id
            tokenizer.hf_tokenizer.pad_token = tokenizer.eos_token
            tokenizer.hf_tokenizer.pad_token_id = tokenizer.eos_token_id
        model_id = pretrained_model_name_or_path
        model_name = kwargs.pop('model_name', None)
        default_chat_template = kwargs.pop('default_chat_template', None)
        self._device = model.device
        self._detokenizer_dir = pretrained_model_name_or_path if tokenizer_name_or_path is None else tokenizer_name_or_path

        super().__init__(model, model_id, tokenizer, model_name, default_chat_template)
    
    @property
    def model(self) -> PreTrainedModel:
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
        gen_kwargs = dict(
            do_sample=(temperature != 0),
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer.hf_tokenizer
            )
        if temperature > 0:
            gen_kwargs['temperature'] = temperature
        if stop:
            kwargs.pop('stop_strings', None)
            gen_kwargs['stop_strings'] = stop
        if top_p is not None:
            gen_kwargs['top_p'] = top_p
        if min_p is not None:
            gen_kwargs['min_p'] = min_p
        if top_k is not None:
            gen_kwargs['top_k'] = top_k
        if repetition_penalty is not None:
            gen_kwargs['repetition_penalty'] = repetition_penalty
        if seed is not None:
            from transformers import set_seed
            set_seed(seed)
        gen_kwargs.update(kwargs)
                
        inputs = self.tokenizer.hf_tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left').to(self._device)
        token_ids = self.model.generate(**inputs, **gen_kwargs)[:, inputs['input_ids'].shape[1]:]
        outputs = self.tokenizer.batch_detokenize(token_ids.cpu().tolist())
        del token_ids
        if self._device.type == 'mps':
            import torch
            torch.mps.empty_cache()
        elif self._device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()

        if stop is not None:
            for s in stop:
                outputs = [o.split(s)[0] for o in outputs]
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
        from threading import Thread
        kwargs.pop('streamer', None)
        streamer = TokenStreamer(self._detokenizer_dir, tokenizer=self.tokenizer.hf_tokenizer)
        stop_ls = [self.tokenizer.eos_token] if (stop is None) else [self.tokenizer.eos_token] + stop
        stop_ls = list(set(stop_ls))
        stop_len = [len(s) for s in stop]
        stop_tuple = list(zip(stop, stop_len))
        stop_tuple.sort(key=lambda x: x[1], reverse=True)


        def process_stop_condition(text: str, 
                                stop: List[str], 
                                stop_condition: StopCondition,
                                is_stop: bool,
                                is_last: bool) -> Tuple[str, str]:
            if is_stop:
                return '', ''
            elif stop_condition.stop_met:
                return text[:-stop_condition.trim_length], '', stop_condition.stop_text
            elif any(sequence_overlap(text, s) for s in stop) and (not is_last):
                return '', text
            else:
                return text, ''

        def call_generate():
            try:
                self.batch_generate(
                    prompts=prompts,
                    stop=stop,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    seed=seed,
                    streamer=streamer,
                    **kwargs
                )
            except Exception as e:
                streamer.end()
                raise e
        trd = Thread(target=call_generate)
        def gen_tokens():
            trd.start()
            texts = [''] * len(prompts)
            is_stopped = [False] * len(prompts)
            for tokens in streamer:
                if not all(is_stopped):
                    texts  = [t + nt for t, nt in zip(texts, tokens)]
                    stop_conditions = [stopping_criteria(t, stop_tuple=stop_tuple) for t in texts]
                    outputs = [process_stop_condition(t, stop_ls, sc, s, False) for t, sc, s in zip(texts, stop_conditions, is_stopped)]
                    is_stopped = [s if s else sc.stop_met for sc, s in zip(stop_conditions, is_stopped)]
                    out_texts, texts = [list(ls) for ls in zip(*outputs)]
                    if not all([ot == '' for ot in out_texts]):
                        yield out_texts
            if not all(is_stopped):
                stop_conditions = [stopping_criteria(t, stop_tuple=stop_tuple) for t in texts]
                outputs = [process_stop_condition(t, stop_ls, sc, s, True) for t, sc, s in zip(texts, stop_conditions, is_stopped)]
                out_texts, texts = [list(ls) for ls in zip(*outputs)]
                if not all([ot == '' for ot in out_texts]):
                    yield out_texts
            trd.join()
        return gen_tokens()

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
        from transformers import LogitsProcessorList
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
        from transformers import LogitsProcessorList
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
        from transformers import LogitsProcessorList
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
        from transformers import LogitsProcessorList
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
    