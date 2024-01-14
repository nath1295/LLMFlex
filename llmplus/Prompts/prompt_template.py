from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal, Union, Tuple

DEFAULT_SYSTEM_MESSAGE = """This is a conversation between a human user and a helpful AI assistant."""

class PromptTemplate:
    """Class for storing prompt format presets.
    """
    def __init__(
            self,
            system_prefix: str,
            system_suffix: str,
            human_prefix: str,
            human_suffix: str,
            ai_prefix: str,
            ai_suffix: str,
            wrapper: List[str],
            stop: Optional[List[str]] = None
    ) -> None:
        """Initialising the chat prompt class.

        Args:
            system_prefix (str): System message prefix.
            system_suffix (str): System message suffix.
            human_prefix (str): User message prefix.
            human_suffix (str): User message suffix.
            ai_prefix (str): Chatbot message prefix.
            ai_suffix (str): Chatbot message suffix.
            wrapper (List[str]): Wrapper for start and end of conversation history.
            stop (Optional[List[str]], optional): List of stop strings for the llm. If None is given, the human_prefix will be used. Defaults to None.
        """
        self._system_prefix = system_prefix
        self._system_suffix = system_suffix
        self._human_prefix = human_prefix
        self._human_suffix = human_suffix
        self._ai_prefix = ai_prefix
        self._ai_suffix = ai_suffix
        self._wrapper = wrapper
        self._stop = stop

    @property
    def system_prefix(self) -> str:
        return self._system_prefix
    
    @property
    def system_suffix(self) -> str:
        return self._system_suffix
    
    @property
    def human_prefix(self) -> str:
        return self._human_prefix
    
    @property
    def human_suffix(self) -> str:
        return self._human_suffix
    
    @property
    def ai_prefix(self) -> str:
        return self._ai_prefix
    
    @property
    def ai_suffix(self) -> str:
        return self._ai_suffix
    
    @property
    def wrapper(self) -> List[str]:
        return self._wrapper

    @property
    def stop(self) -> List[str]:
        return self._stop if isinstance(self._stop, list) else [self.human_prefix, f'{self.ai_suffix}{self.human_prefix}']
    
    @property
    def template_name(self) -> str:
        """Name of the template.

        Returns:
            str: Name of the template.
        """
        return getattr(self, '_template_name', 'Unititled template')

    def format_history(self, history: Union[List[str], List[Tuple[str, str]]], use_wrapper: bool = True) -> str:
        """Formatting a list of conversation history into a full string of conversation history.

        Args:
            history (Union[List[str], List[Tuple[str, str]]]): List of conversation history. 
            use_wrapper (bool, optional): Whether to format the conversation history with the wrappers. Defaults to True.

        Returns:
            str: Full string of conversation history.
        """
        if len(history) == 0:
            return ''
        elif not isinstance(history[0], str):
            body = list(map(lambda x: f'{self.human_prefix}{x[0]}{self.human_suffix}{self.ai_prefix}{x[1]}{self.ai_suffix}', history))
            body = ''.join(body)
            if use_wrapper:
                body = self.wrapper[0] + body.removeprefix(self.human_prefix)
                body = body.removesuffix(self.ai_suffix) + self.wrapper[1]
        else:
            length = len(history)
            is_even = length % 2 == 0
            lead = history[0]
            alt_history = history if is_even else history[1:]
            alt_history = list(map(lambda x: (alt_history[x * 2], alt_history[x * 2 + 1]), range(len(alt_history) // 2)))
            body = list(map(lambda x: f'{self.human_prefix}{x[0]}{self.human_suffix}{self.ai_prefix}{x[1]}{self.ai_suffix}', alt_history))
            body = ''.join(body)
            if not is_even:
                body = f'{self.ai_prefix}{lead}{self.ai_suffix}' + body
            if use_wrapper:
                if is_even:
                    body = self.wrapper[0] + body.removeprefix(self.human_prefix)
                    body = body.removesuffix(self.ai_suffix) + self.wrapper[1]
                else:
                    body = body.removesuffix(self.ai_suffix) + self.wrapper[1]
        return body

    def create_prompt(self, user: str, system: str = DEFAULT_SYSTEM_MESSAGE, history: Union[List[str], List[Tuple[str, str]]] = []) -> str:
        """Creating the full chat prompt.

        Args:
            user (str): Latest user input.
            system (str, optional): System message. Defaults to DEFAULT_SYSTEM_MESSAGE.
            history (Union[List[str], List[Tuple[str, str]]], optional): List of conversation history. Defaults to [].

        Returns:
            str: The full prompt.
        """
        head = f'{self.system_prefix}{system}{self.system_suffix}' if system.strip(' \n\r\t') != '' else ''
        body = self.format_history(history=history, use_wrapper=(head==''))
        tail = f'{self.human_prefix}{user}{self.human_suffix}{self.ai_prefix}'
        prompt = head + body + tail
        return prompt
    
    @classmethod
    def from_dict(cls, format_dict: Dict[str, Any], template_name: Optional[str] = None) -> PromptTemplate:
        """Initialise the prompt template from a dictionary.

        Args:
            format_dict (Dict[str, Any]): Dictionary of the prompt format.
            template_name (Optional[str], optional): Name of the template. Defaults to None.

        Returns:
            PromptTemplate: The initialised PromptTemplate instance.
        """
        template = cls(**format_dict)
        if template_name is not None:
            template._template_name = template_name
        return template
    
    @classmethod
    def from_json(cls, file_dir: str) -> PromptTemplate:
        """Initialise the prompt template from a json file.

        Args:
            file_dir (str): json file path of the prompt format.

        Returns:
            PromptTemplatet: The initialised PromptTemplate instance.
        """
        from ..utils import read_json
        return cls.from_dict(read_json(file_dir=file_dir), template_name=file_dir)
    
    @classmethod
    def from_preset(cls, style: Literal['Default', 'Default Instruct', 'Llama2', 'Vicuna', 'ChatML', 'Zephyr', 'OpenChat']) -> PromptTemplate:
        """Initialise the prompt template from a preset.

        Args:
            style (Literal[&#39;Default&#39;, &#39;Default Instruct&#39;, &#39;Llama2&#39;, &#39;Vicuna&#39;, &#39;ChatML&#39;, &#39;Zephyr&#39;, &#39;OpenChat&#39;]): Format of the prompt.

        Returns:
            PromptTemplate: The initialised PromptTemplate instance.
        """
        return cls.from_dict(presets[style], template_name=style)
    
    def to_dict(self, return_raw_stop: bool = True) -> Dict[str, Any]:
        """Export the class as a dictionary.

        Args:
            return_raw_stop (bool, optional): Whether to return the stop list or the raw input stop value of the PromptTemplate instance.

        Returns:
            Dict[str, Any]: Prompt format as a dictionary.
        """
        return dict(
            system_prefix = self.system_prefix,
            system_suffix = self.system_suffix,
            human_prefix = self.human_prefix,
            human_suffix = self.human_suffix,
            ai_prefix = self.ai_prefix,
            ai_suffix = self.ai_suffix,
            stop = self._stop if return_raw_stop else self.stop
        )
    
presets = {
    'Default' : {
        'system_prefix': 'SYSTEM:\n',
        'system_suffix': '\n\nCurrent conversation:\n',
        'human_prefix': 'USER: ',
        'human_suffix': '\n',
        'ai_prefix': 'ASSISTANT: ',
        'ai_suffix': '\n',
        'wrapper': ['USER: ', '\n'],
        'stop': None
    },
    'Default Instruct' : {
        'system_prefix': 'SYSTEM:\n',
        'system_suffix': '\n\n',
        'human_prefix': 'USER: ',
        'human_suffix': '\n',
        'ai_prefix': 'ASSISTANT: ',
        'ai_suffix': '\n',
        'wrapper': ['USER: ', '\n'],
        'stop': None
    },
    'Llama2' : {
        'system_prefix': '[INST] <<SYS>>\n',
        'system_suffix': '\n<</SYS>>\n\n',
        'human_prefix': '',
        'human_suffix': ' [/INST] ',
        'ai_prefix': '',
        'ai_suffix': ' </s><s>[INST] ',
        'wrapper': ['<s>[INST] ', ' </s>'],
        'stop': ['</s>', '</s><s>', '[INST]', '<s>[INST]']
    },
    'Vicuna' : {
        'system_prefix': '',
        'system_suffix': '\n\n',
        'human_prefix': 'USER: ',
        'human_suffix': '\n',
        'ai_prefix': 'ASSISTANT: ',
        'ai_suffix': '\n',
        'wrapper': ['USER: ', '\n'],
        'stop': None
    },
    'ChatML' : {
        'system_prefix': '<|im_start|>system\n',
        'system_suffix': '<|im_end|>\n',
        'human_prefix': '<|im_start|>user\n',
        'human_suffix': '<|im_end|>\n',
        'ai_prefix': '<|im_start|>assistant\n',
        'ai_suffix': '<|im_end|>\n',
        'wrapper': ['<|im_start|>user\n', '<|im_end|>\n'],
        'stop': ['<|im_start|>', '<|im_end|>', '<|im_start|>user\n', '<|im_end|>\n<|im_start|>user\n']
    },
    'Zephyr' : {
        'system_prefix': '<|system|>\n',
        'system_suffix': '</s>\n',
        'human_prefix': '<|user|>\n',
        'human_suffix': '</s>\n',
        'ai_prefix': '<|assistant|>\n',
        'ai_suffix': '</s>\n',
        'wrapper': ['<|user|>\n', '</s>\n'],
        'stop': ['</s>', '</s>\n', '<|user|>\n', '</s>\n<|user|>\n']
    },
    'OpenChat' : {
        'system_prefix': 'GPT4 Correct System: ',
        'system_suffix': '<|end_of_turn|>',
        'human_prefix': 'GPT4 Correct User: ',
        'human_suffix': '<|end_of_turn|>',
        'ai_prefix': 'GPT4 Correct Assistant: ',
        'ai_suffix': '<|end_of_turn|>',
        'wrapper': ['GPT4 Correct User: ', '<|end_of_turn|>'],
        'stop': ['</s>', '<|end_of_turn|>', '<|end_of_turn|>GPT4 Correct Assistant: ']
    },
}