from __future__ import annotations
from jinja2 import Environment
from typing import List, Dict, Any, Optional, Literal, Union, Tuple

DEFAULT_SYSTEM_MESSAGE = """This is a conversation between a human user and a helpful AI assistant."""

presets = {
    'Default' : {
        'template': "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{{ 'SYSTEM:\n' + messages[0]['content'].strip() + '\n\n' }}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'].strip() + '\n' }}{% else %}{{ message['role'].upper() + ': ' + message['content'].strip() + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
        'eos_token': '</s>',
        'bos_token': '<s>',
        'stop': ['\nASSISTANT', '\nUSER:', 'ASSISTANT:', 'USER:']
    },
    'Llama2' : {
        'template': "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
        'eos_token': '</s>',
        'bos_token': '<s>',
        'stop': ['</s>', '</s><s>', '[INST]', '<s>[INST]'],
        'keywords': ['[INST]', '[/INST]']
    },
    'Llama3' : {
        'template': "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% else %}{{ eos_token }}{% endif %}",
        'eos_token': '<|end_of_text|>',
        'bos_token': '<|begin_of_text|>',
        'stop': ['</s>', '<|end_of_text|>', '<|eot_id|>', '<|start_header_id|>'],
        'keywords': ['<|start_header_id|>', '<|end_header_id|>']
    },
    'Vicuna' : {
        'template': "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{{ messages[0]['content'].strip() + '\n\n' }}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
        'eos_token': '</s>',
        'bos_token': '<s>',
        'stop': ['\nASSISTANT', '\nUSER:', 'ASSISTANT:', 'USER:', '</s>'],
        'keywords': ['USER:', 'ASSISTANT:']
    },
    'ChatML' : {
        'template': "{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        'eos_token': '<|im_end|>',
        'bos_token': '<s>',
        'stop': ['<|im_start|>', '<|im_end|>', '<|im_start|>user\n', '<|im_end|>\n<|im_start|>user\n', '</s>'],
        'keywords': ['<|im_start|>', '<|im_end|>']
    },
    'Zephyr' : {
        'template': "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
        'eos_token': '</s>',
        'bos_token': '<s>',
        'stop': ['</s>', '</s>\n', '<|user|>\n', '</s>\n<|user|>\n'],
        'keywords': ['<|system|>']
    },
    'OpenChat' : {
        'template': "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",
        'eos_token': '<|end_of_turn|>',
        'bos_token': '<s>',
        'stop': ['</s>', '<|end_of_turn|>', '<|end_of_turn|>GPT4 Correct Assistant: '],
        'keywords': ['GPT4 Correct']
    },
    'Alpaca' : {
        'template': "{% for message in messages %}{% if message['role'] == 'system' %}{% if message['content']%}{{'### Instruction: ' + message['content']+'\n'}}{% endif %}{% elif message['role'] == 'user' %}{{'### Input: ' + message['content']+'\n'}}{% elif message['role'] == 'assistant' %}{{'### Response: '  + message['content'] + '\n'}}{% endif %}{% if loop.last and add_generation_prompt %}{{ '### Response:' }}{% endif %}{% endfor %}",
        'eos_token': '<|end_of_turn|>',
        'bos_token': '<s>',
        'stop': ['### Input: ', '\n### Input: ', '###'],
        'keywords': ['### Input:', '### Response:']
    },
    'Phi3' : {
        'template': "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
        'eos_token': '<|endoftext|>',
        'bos_token': '<s>',
        'stop': ['<|end|>', '<|endoftext|>'],
        'keywords': ['<|end|>']
    },
}

hidden_presets = {
    'Llama2' : {
        'template': "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{% if loop.index0 == 0 %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% else %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% endif %}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{% if loop.index0 == 0 and system_message != false %}{% set prefix = '[INST]' + '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + '[/INST] ' %}{% set content = message['content'] %}{% elif loop.index0 == 0 %}{% set prefix = '[INST][/INST] ' %}{% set content = message['content'] %}{% else %}{% set prefix = ' ' %}{% endif %}{{ prefix  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
        'eos_token': '</s>',
        'bos_token': '<s>',
        'stop': ['</s>', '</s><s>', '[INST]', '<s>[INST]']
    },
    'Zephyr' : {
        'template': "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% else  %}\n{{ '<|' + message['role'] + '|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
        'eos_token': '</s>',
        'bos_token': '<s>',
        'stop': ['</s>', '</s>\n', '<|user|>\n', '</s>\n<|user|>\n']
    },
    'Vicuna' : {
        'template': "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{{ messages[0]['content'].strip() + '\n\n' }}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}{% else %}{{ message['role'].upper() + ': ' + message['content'].strip() + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
        'eos_token': '</s>',
        'bos_token': '<s>',
        'stop': ['\nASSISTANT', '\nUSER:', 'ASSISTANT:', 'USER:', '</s>']
    },
    'Phi3' : {
        'template': "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}",
        'eos_token': '<|endoftext|>',
        'bos_token': '<s>',
        'stop': ['<|end|>', '<|endoftext|>'],
        'keywords': ['<|end|>']
    },
}

PRESET_FORMATS =  Literal['Default', 'Llama2', 'Llama3', 'Vicuna', 'ChatML', 'Zephyr', 'OpenChat', 'Alpaca', 'Phi3']

class PromptTemplate:
    """Class for storing prompt format presets.
    """
    def __init__(
            self,
            template: str,
            eos_token: Optional[str],
            bos_token: Optional[str],
            stop: Optional[List[str]] = None,
            force_real_template: bool = False,
            **kwargs
    ) -> None:
        """Initialising the chat prompt class.

        Args:
            template (str): Jinja2 template.
            eos_token (Optional[str]): EOS token string.
            bos_token (Optional[str]): BOS token string.
            stop (Optional[List[str]], optional): List of stop strings for the llm. If None is given, the EOS token string will be used. Defaults to None.
            force_real_template (bool, optional): Whether to render the given template. For most templates it has no effects. Only for some restrictive templates like llama2. Defaults to False.
        """
        self._template = template
        self._eos_token = eos_token
        self._bos_token = bos_token
        self._stop = stop
        self._force_real_template = force_real_template
        self._keywords = kwargs.get('keywords', [])

    @property
    def template(self) -> str:
        """Jinja template string.

        Returns:
            str: Jinja template string.
        """
        return self._template
    
    @property
    def _hidden_template(self) -> str:
        """To fix issues with some default templates like llama 2.

        Returns:
            str: The actual template to be rendered.
        """
        config = hidden_presets.get(self.template_name)
        return self.template if config is None else config['template']
    
    @property
    def rendered_template(self) -> Environment:
        """Rendered Jinja template.

        Returns:
            Environment: Rendered Jinja template.
        """
        if not hasattr(self, '_rendered_template'):
            from jinja2 import BaseLoader
            template = self.template if self._force_real_template else self._hidden_template
            self._rendered_template = Environment(loader=BaseLoader).from_string(template)
        return self._rendered_template
    
    @property
    def keywords(self) -> List[str]:
        """List of keywords to search for in Jinja templates for template detection. Used for presets.

        Returns:
            List[str]: List of keywords to search for in Jinja templates for template detection.
        """
        if not hasattr(self, '_keywords'):
            self._keywords = []
        return self._keywords

    @property
    def eos_token(self) -> Optional[str]:
        return self._eos_token
    
    @property
    def bos_token(self) -> Optional[str]:
        return self._bos_token

    @property
    def stop(self) -> List[str]:
        return self._stop if isinstance(self._stop, list) else [self.eos_token] if self.eos_token is not None else []
    
    @property
    def template_name(self) -> str:
        """Name of the template.

        Returns:
            str: Name of the template.
        """
        if not hasattr(self, '_template_name'):
            self._template_name = 'Unititled template'
            for k, v in presets.items():
                if self.template == v['template']:
                    self._template_name = k
                    break
        return self._template_name
    
    @property
    def allow_custom_role(self) -> bool:
        """Check if custom role can be used with the prompt template.

        Returns:
            bool: Whether custom role can be used with the prompt template.
        """
        if not hasattr(self, '_allow_custom_role'):
            test = [dict(role='user', content='random message'), dict(role='customrole', content='exist')]
            try: 
                output = self.create_custom_prompt(test)
                self._allow_custom_role = 'customrole' in output.lower()
            except:
                self._allow_custom_role = False
        return self._allow_custom_role



    def format_history(self, history: Union[List[str], List[Tuple[str, str]]], return_list: bool = False) -> Union[str, List[Dict[str, str]]]:
        """Formatting a list of conversation history into a full string of conversation history or a list of messages for the Jinja template to render.

        Args:
            history (Union[List[str], List[Tuple[str, str]]]): List of conversation history. 
            return_list (bool, optional): Whether to return a list of messages for the Jinja template to render. Defaults to False.

        Returns:
            Union[str, List[Dict[str, str]]]: A full string of conversation history or a list of messages for the Jinja template to render.
        """
        if len(history) == 0:
            return [] if return_list else ''
        elif not isinstance(history[0], str):
            body = list(map(lambda x: [dict(role='user', content=x[0]), dict(role='assistant', content=x[1])], history))
            body = sum(body, [])
        else:
            length = len(history)
            is_even = length % 2 == 0
            half = int(length / 2) if is_even else int((length + 1) / 2)
            roles = ['user', 'assistant'] * half
            roles = roles if is_even else roles[1:]
            body = list(map(lambda x: dict(role=x[0], content=x[1]), list(zip(roles, history))))
        if return_list:
            return body
        return self.rendered_template.render(messages=body, bos_token=self.bos_token, eos_token=self.eos_token, add_generation_prompt=False)

    def create_prompt(self, user: str, system: str = DEFAULT_SYSTEM_MESSAGE, history: Optional[Union[List[str], List[Tuple[str, str]]]] = None) -> str:
        """Creating the full chat prompt.

        Args:
            user (str): Latest user input.
            system (str, optional): System message. Defaults to DEFAULT_SYSTEM_MESSAGE.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): List of conversation history. Defaults to None.

        Returns:
            str: The full prompt.
        """
        head = [dict(role='system', content=system)] if system.strip() != '' else []
        history = [] if history is None else history
        body = self.format_history(history=history, return_list=True)
        tail = [dict(role='user', content=user)]
        prompt = head + body + tail
        return self.rendered_template.render(messages=prompt, bos_token=self.bos_token, eos_token=self.eos_token, add_generation_prompt=True)
    
    def create_custom_prompt(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Creating a custom prompt with your given list of messages. Each message should contain a dictionary with the key "role" and "content".

        Args:
            messages (List[Dict[str, str]]): List of messages. Each message should contain a dictionary with the key "role" and "content".
            add_generation_prompt (bool, optional): Whether to add the assistant tokens at the end of the prompt. Defaults to True.

        Returns:
            str: The full prompt given your messages.
        """
        return self.rendered_template.render(messages=messages, bos_token=self.bos_token, eos_token=self.eos_token, add_generation_prompt=add_generation_prompt)
    
    def create_custom_prompt_with_open_role(self, messages: List[Dict[str, str]], end_role: str = '', begin_text: str = '') -> str:
        """Creating a custom prompt with your given list of messages. Each message should contain a dictionary with the key "role" and "content". The prompt will end with starting prompt of the end_role instead of assistant.

        Args:
            messages (List[Dict[str, str]]): List of messages. Each message should contain a dictionary with the key "role" and "content".
            end_role (str, optional): The role for text generation instead of assistant. If an empty string is given, it means that the role can be anything the llm is going to generate. Defaults to ''.
            begin_text (str, optional): The beginning text of the last role. Defaults to ''.

        Returns:
            str: The full prompt with your custom role.
        """
        if self.allow_custom_role:
            role_ph = end_role if end_role.strip() != '' else 'place_holder_role'
            text_ph = '$$$PLACE_HOLDER_TEXT$$$'
            prompt = self.create_custom_prompt(messages=messages + [dict(role=role_ph, content=text_ph)], add_generation_prompt=False)
            prompt = text_ph.join(prompt.split(text_ph)[:-1])
            real = self.create_custom_prompt(messages=messages + [dict(role=end_role, content=text_ph)], add_generation_prompt=False)
            real = text_ph.join(real.split(text_ph)[:-1])
            if real == prompt:
                return real + begin_text
            elif begin_text.strip() != '':
                return real + begin_text
            else:
                import os
                return os.path.commonprefix([real, prompt])
        else:
            raise AssertionError(f'This prompt template "{self.template_name}" does not allow custom_roles.')

    
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
    def from_preset(cls, style: PRESET_FORMATS, force_real_template: bool = False) -> PromptTemplate:
        """Initialise the prompt template from a preset.

        Args:
            style (PRESET_FORMATS): Format of the prompt.
            force_real_template (bool, optional): Whether to render the given template. For most templates it has no effects. Only for some restrictive templates like llama2. Defaults to False.

        Returns:
            PromptTemplate: The initialised PromptTemplate instance.
        """
        from copy import deepcopy
        preset = deepcopy(presets[style])
        preset['force_real_template'] = force_real_template
        return cls.from_dict(preset, template_name=style)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export the class as a dictionary.

        Returns:
            Dict[str, Any]: Prompt format as a dictionary.
        """
        return dict(
            template = self.template,
            eos_token = self.eos_token,
            bos_token = self.bos_token,
            stop = self.stop if self.stop is not None else [self.eos_token]
        )
