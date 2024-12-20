from .base_engine import BaseEngine
from typing import Optional, Any, List, Dict, Iterator, Literal, Union

class MLXEngine(BaseEngine):
    """Class for LLM engine using MLX.
    """
    def __init__(self, 
            pretrained_model_name_or_path: str,
            quantization: Literal['fp16', 'q8', 'q4', 'q2'] = 'fp16',
            tokenizer_name_or_path: Optional[str] = None,
            revision: Optional[str] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            prefill_step_size: int = 512,
            **kwargs) -> None:
        """Initializes the MLXEngine with the given parameters.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
            quantization (Literal['fp16', 'q8', 'q4', 'q2'], optional): The quantization method to use for the model. Defaults to 'fp16'.
            tokenizer_name_or_path (Optional[str], optional): The name or path of the tokenizer to use. Defaults to None.
            revision (Optional[str], optional): The branch of the Huggingface repository to use. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
            prefill_step_size (int, optional): The batch size of prompt processing. Defaults to 512.
        """
        try:
            from mlx_textgen.engine import ModelEngine, ModelConfig
        except:
            raise ModuleNotFoundError('"mlx-textgen" is not installed. To use "MLXEngine", please install "mlx-textgen" by running `pip install mlx-textgen`.')
        from ...Tokenizer.huggingface_tokenizer import HuggingFaceTokenizer
        config = ModelConfig(
            model_id_or_path=pretrained_model_name_or_path,
            tokenizer_id_or_path=tokenizer_name_or_path,
            quant=quantization,
            revision=revision,
            model_config=model_kwargs,
            tokenizer_config=tokenizer_kwargs
        )
        model = ModelEngine(models=config, verbose=False, prefill_step_size=prefill_step_size)
        model_id = list(model.models.keys())[0]
        model._switch_model(model_name=model_id)
        tokenizer = HuggingFaceTokenizer.from_hf_tokenizer(model.tokenizer._tokenizer)
        model_name = kwargs.get('model_name', None)
        default_chat_template = kwargs.pop('default_chat_template', None)
        super().__init__(model, model_id, tokenizer, model_name, default_chat_template)

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
        outputs = self.model.generate(
            model=self.model_id,
            prompt=prompts,
            completion_type='text_completion',
            stream=False,
            stop=stop,
            max_tokens=max_new_tokens,
            n=1,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            min_p=min_p,
            seed=seed,
            **kwargs
        )['choices']
        outputs = [o['text'] for o in outputs]
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
        outputs = self.model.generate(
            model=self.model_id,
            prompt=prompts,
            completion_type='text_completion',
            stream=True,
            stop=stop,
            max_tokens=max_new_tokens,
            n=1,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            min_p=min_p,
            seed=seed,
            **kwargs
        )
        num_prompts = len(prompts)
        def _gen_tokens():
            for i in outputs:
                outs = [''] * num_prompts
                index = i['choices'][0]['index']
                outs[index] = i['choices'][0]['text']
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
            seed=seed,
            guided_json=json_schema
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
            guided_choice=choices
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
            guided_regex=regex
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
            guided_grammar=cfg_grammar
        )
        arg_dict.update(kwargs)
        if stream:
            return self.batch_generate_stream(**arg_dict)
        else:
            return self.batch_generate(**arg_dict)
    
    def unload(self) -> None:
        import mlx.core as mx
        del self._model, self._tokenizer
        mx.metal.clear_cache()