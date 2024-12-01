from ..Tokenizer.base_tokenizer import BaseTokenizer
from .presets import PRESETS, MSG_SINGLE_ASSISTANT, MSG_WITH_TOOL, TOOL_LIST, DEFAULT_TOOL_SYSTEM
from pydantic import BaseModel
import json
from typing import List, Optional, Dict, Any, Union, Literal

class ToolCallContent(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCall(BaseModel):
    function: ToolCallContent

class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Optional[Any] = None
    tool_call: Optional[List[ToolCall]] = None

def get_chat_template(
        chat_template: Optional[Literal['chatml', 'llama3', 'mistral', 'gemma', 'deepseek', 'openchat', 'phi']] = None, 
        tokenizer: Optional[BaseTokenizer] = None, 
        tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """Retrieve the chat template based on the provided key, tokenizer, and tool availability.

    This function returns the appropriate chat template based on the given `chat_template` key, `tokenizer`, and the existence of `tools`.

    Args:
        chat_template (Optional[Literal['chatml', 'llama3', 'mistral', 'gemma', 'deepseek', 'openchat', 'phi']], optional): The key to the chat template to use. Defaults to None.
        tokenizer (Optional[BaseTokenizer], optional): The tokenizer that provides information about the chat template. Defaults to None.
        tools (Optional[List[Dict[str, Any]]], optional): The list of tools to use in the template. Defaults to None.

    Returns:
        str: The Jinja chat template.
    """
    if (chat_template is None) and (tokenizer is None):
        raise ValueError('Must provide at least one of "chat_template" and "tokenizer".')
    if chat_template in PRESETS.keys():
        return PRESETS[chat_template]
    elif chat_template is not None:
        accepted_templates = '"' + '", "'.join(list(PRESETS.keys())) + '"'
        raise ValueError(f'"chat_template" must be one of the followings: {accepted_templates}')
    elif tokenizer.__class__.__name__ == 'HuggingFaceTokenizer':
        template = tokenizer.hf_tokenizer.chat_template
        if isinstance(template, dict):
            if tools is not None and "tool_use" in template:
                chat_template = template["tool_use"]
            elif "default" in template:
                chat_template = template["default"]
            else:
                raise ValueError(
                    "Cannot find chat template for the tokenizer."
                )
            return chat_template
        elif isinstance(template, str):
            return template
    elif tokenizer.__class__.__name__ == 'LlamaCppTokenizer':
        template_key = tokenizer.llama_tokenizer.chat_format
        if template_key in PRESETS.keys():
            return PRESETS[template_key]
        elif template_key == 'llama2':
            return PRESETS['mistral']
        else:
            return PRESETS['chatml']
    elif tokenizer.__class__.__name__ == 'OpenAITokenizer':
        return PRESETS['openchat']
    else:
        return PRESETS['chatml']
        
class ChatTemplate:
    """A class to manage chat templates for conversational AI.

    This class provides a way to retrieve chat templates based on the given chat template key, tokenizer, and the existence of tools.
    It also handles the case where the tokenizer is a Hugging Face tokenizer and extracts the appropriate chat template from its configuration.
    """
    def __init__(self, 
            tokenizer: BaseTokenizer, 
            chat_template: Optional[Literal['chatml', 'llama3', 'mistral', 'gemma', 'deepseek', 'openchat', 'phi']] = None
        ) -> None:
        """Initialize the ChatTemplate instance with the provided tokenizer and chat template key.
        
        Args:
            tokenizer (BaseTokenizer): The tokenizer that provides information about the chat template.
            chat_template (Optional[Literal['chatml', 'llama3', 'mistral', 'gemma', 'deepseek', 'openchat', 'phi']], optional): The key to the chat template to use. Defaults to None.
        """
        self._tokenizer = tokenizer
        self._chat_template = chat_template

    @property
    def tokenizer(self) -> BaseTokenizer:
        """Gets the tokenizer associated with this chat template.

        Returns:
            BaseTokenizer: The tokenizer associated with this chat template.
        """
        return self._tokenizer
    
    @property
    def support_system(self) -> bool:
        """Checks if the chat template supports system message.

        Returns:
            bool: True if the chat template supports system message, False otherwise.
        """
        if not hasattr(self, '_support_system'):
            try:
                system = 'Test system message, see if exist.'
                messages = [dict(role='system', content=system), dict(role='user', content='Hi there')]
                prompt = self._apply_chat_template(messages=messages)
                self._support_system = system in prompt
            except:
                self._support_system = False
        return self._support_system
    
    @property
    def support_tool_call(self) -> bool:
        """Checks if the chat template supports tool calls.

        Returns:
            bool: True if the chat template supports tool calls, False otherwise.
        """
        if not hasattr(self, '_support_tool_call'):
            try:
                p_with_tool = self._apply_chat_template(MSG_WITH_TOOL[:1], tools=TOOL_LIST, add_generation_prompt=True)
                p_wo_tool = self._apply_chat_template(MSG_WITH_TOOL[:1], add_generation_prompt=True)
                self._support_tool_call = p_with_tool != p_wo_tool
            except:
                self._support_tool_call = False
        return self._support_tool_call
    
    @property
    def allow_multiple_assistant(self) -> bool:
        """Checks if the chat template allows multiple consecutive assistant messages.

        Returns:
            bool: True if the chat template allows multiple consecutive assistant messages, False otherwise.
        """
        if not hasattr(self, '_allow_multiple_assistant'):
            try:
                self._apply_chat_template(MSG_SINGLE_ASSISTANT * 2, add_generation_prompt=False, continue_final_message=True)
                self._allow_multiple_assistant = True
            except:
                self._allow_multiple_assistant = False
        return self._allow_multiple_assistant

    @property
    def tool_start(self) -> str:
        """Gets the start string for tool calls in the chat template.

        Returns:
            str: The start string for tool calls in the chat template.
        """
        if not hasattr(self, '_tool_start'):
            if self.support_tool_call:
                p_with_tool = self._apply_chat_template(MSG_WITH_TOOL, tools=TOOL_LIST, add_generation_prompt=True)
                p_wo_tool = self._apply_chat_template(MSG_WITH_TOOL[:1], tools=TOOL_LIST, add_generation_prompt=False)
                diff_str = p_with_tool.removeprefix(p_wo_tool)
                tool_first_index = diff_str.find('{')
                self._tool_start = diff_str[:tool_first_index]
            else:
                self._tool_start = '<tool_call>\n'
        return self._tool_start
    
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validates the given messages to ensure they are in the correct format and contain the necessary fields.

        Args:
            messages (List[Dict[str, Any]]): The list of messages to validate.

        Raises:
            Exception: If any of the messages are not in the correct format or do not contain the necessary fields.

        Returns:
            List[Dict[str, Any]]: The validated list of messages.
        """
        for msg in messages:
            ChatMessage(**msg)
        if len(messages) == 0:
            raise Exception(f'Must have at least one message.')
        return messages
    
    def _validate_message_seq(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validates the sequence of messages to ensure they are in the correct order and contain the necessary fields.

        Args:
            messages (List[Dict[str, Any]]): The list of messages to validate.

        Raises:
            Exception: If any of the messages are not in the correct order or do not contain the necessary fields.

        Returns:
            List[Dict[str, Any]]: The validated list of messages.
        """
        if self.allow_multiple_assistant and self.support_tool_call:
            new_messages = messages
        elif self.allow_multiple_assistant and (not self.support_tool_call):
            new_messages = []
            for msg in messages:
                if (msg['role'] == 'assistant') and msg.get('tool_call'):
                    tool_call = '\n'.join(['<tool_call>\n' + json.dumps(tc) + '\n</tool_call>' for tc in msg['tool_call']])
                    new_messages.append(dict(role='assistant', content=msg.get('content', '') + tool_call))
                elif (msg['role'] == 'tool'):
                    content = msg.get('content', '')
                    content = content if isinstance(content, str) else json.dumps(content)
                    content = '<tool_response>\n' + content + '\n</tool_response>'
                    new_messages.append(dict(role='user', content=content))
                else:
                    new_messages.append(msg)
        elif self.support_tool_call:
            new_messages = []
            last_role = None
            for msg in messages:
                if (msg['role'] == 'assistant') and last_role != 'user':
                    new_messages.extend([dict(role='user', content=''), msg])
                else:
                    new_messages.append(msg)
                last_role = msg['role']
        else:
            new_messages = []
            last_role = None
            for msg in messages:
                if (msg['role'] == 'assistant') and msg.get('tool_call'):
                    tool_call = '\n'.join(['<tool_call>\n' + json.dumps(tc) + '\n</tool_call>' for tc in msg['tool_call']])
                    to_append = dict(role='assistant', content=msg.get('content', '') + tool_call)
                elif (msg['role'] == 'tool'):
                    content = msg.get('content', '')
                    content = content if isinstance(content, str) else json.dumps(content)
                    content = '<tool_response>\n' + content + '\n</tool_response>'
                    to_append = dict(role='user', content=content)
                else:
                    to_append = msg
                if (to_append['role'] == 'assistant') and last_role != 'user':
                    new_messages.extend([dict(role='user', content=''), to_append])
                else:
                    new_messages.append(to_append)
                last_role = to_append['role']
        if not self.support_system and (new_messages[0]['role'] == 'system'):
            system = new_messages[0]['content']
            new_messages = new_messages[1:]
            if (len(new_messages) > 0) and (new_messages[0]['role'] == 'user'):
                first_message = new_messages[0]['content']
                first_message = '<system>\n' + system + '\n</system>\n\n' + first_message
                new_messages[0]['content'] = first_message
            else:
                raise Exception(f'First message after the system message is not a user message.')
        return new_messages

    def _apply_chat_template(self, 
            messages: List[Dict[str, Any]], 
            tools: Optional[List[Dict[str, Any]]] = None, 
            add_generation_prompt: bool = True,
            continue_final_message: bool = False
        ) -> str:
        """Applies the chat template to the given messages and tools.

        This method takes a list of messages and an optional list of tools, and applies the chat template to generate a string
        representing the conversation. If tools are provided, the method will include the information about the tools in the system prompt.
        If `add_generation_prompt` is set to True, the method will add a generation prompt to the end of the conversation string.

        Args:
            messages (List[Dict[str, Any]]): The list of messages to apply the chat template to.
            tools (Optional[List[Dict[str, Any]]], optional): The list of tools to include in the conversation. Defaults to None.
            add_generation_prompt (bool, optional): Whether to add a generation prompt to the end of the conversation string. Defaults to True.
            continue_final_message (bool, optional): Whether to continue the final message in the conversation string. Defaults to False.

        Returns:
            str: The conversation string generated by applying the chat template to the given messages and tools.
        """
        from jinja2 import Environment, BaseLoader
        from copy import deepcopy
        template = Environment(loader=BaseLoader).from_string(get_chat_template(self._chat_template, tokenizer=self._tokenizer, tools=tools))
        prompt = template.render(
            messages=deepcopy(messages), 
            tools=tools, 
            add_generation_prompt=add_generation_prompt,
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token
        )
        if continue_final_message:
            final_message = messages[-1]["content"]
            if isinstance(final_message, (list, tuple)):
                final_message = final_message[-1]["text"]
            try:
                prompt = prompt[: prompt.rindex(final_message) + len(prompt)]
            except:  # noqa: E722
                # Some chat templates like Llama-3.1 trim messages before rendering, so we must do the same here.
                final_message = final_message.strip()
                prompt = prompt[: prompt.rindex(final_message) + len(prompt)]
        return prompt
    
    def apply_chat_template(self, 
            messages: List[Dict[str, Any]], 
            tools: Optional[List[Dict[str, Any]]] = None, 
            tool_choice: Union[Literal['none', 'auto', 'required'], Dict[str, Union[str, Dict[str, str]]]] = 'auto',
            add_generation_prompt: bool = True,
            continue_final_message: bool = False
            ) -> str:
        """Applies the chat template to the given messages and tools, considering tool choice.

        This method takes a list of messages, an optional list of tools, and a tool choice parameter, and applies the chat template to generate a string
        representing the conversation. If tools are provided, the method will include the information about the tools in the system prompt.
        The `tool_choice` parameter determines how tools are used in the conversation. If set to 'none', no tools will be used. If set to 'auto',
        tools will be used if the assistant's response includes a tool call. If set to 'required', at least one tool must be used in the conversation.
        If `add_generation_prompt` is set to True, the method will add a generation prompt to the end of the conversation string.

        Args:
            messages (List[Dict[str, Any]]): The list of messages to apply the chat template to.
            tools (Optional[List[Dict[str, Any]]], optional): The list of tools to include in the conversation. Defaults to None.
            tool_choice (Union[Literal['none', 'auto', 'required'], Dict[str, Union[str, Dict[str, str]]]], optional): The tool choice parameter. Defaults to 'auto'.
            add_generation_prompt (bool, optional): Whether to add a generation prompt to the end of the conversation string. Defaults to True.
            continue_final_message (bool, optional): Whether to continue the final message in the conversation string. Defaults to False.

        Returns:
            str: The conversation string generated by applying the chat template to the given messages and tools, considering tool choice.
        """
        from copy import deepcopy
        messages = self._validate_messages(messages=deepcopy(messages))
        messages = self._validate_message_seq(messages=messages)
        tools = tools if tool_choice != 'none' else None
        if tools and (not self.support_tool_call):
            tool_json = '\n'.join([json.dumps(tool) for tool in tools])
            if messages[0]['role'] == 'system':
                content = messages[0].get('content', '') + DEFAULT_TOOL_SYSTEM.replace('$$tool_list$$', tool_json)
                messages[0]['content'] = content.strip()
            else:
                messages = [dict(role='system', content=DEFAULT_TOOL_SYSTEM.replace('$$tool_list$$', tool_json).strip())] + messages
        if (messages[-1]['role'] == 'assistant') and (messages[-1]['content']) and continue_final_message:
            add_generation_prompt = False
        prompt = self._apply_chat_template(messages=messages, 
            tools=tools if self.support_tool_call else None, 
            add_generation_prompt=add_generation_prompt, 
            continue_final_message=continue_final_message)
        if tool_choice == 'required':
            prompt += self.tool_start + '{"function": {' + '"name": "'
        elif isinstance(tool_choice, dict):
            tool_name = tool_choice['function']['name']
            prompt += self.tool_start + '{"function": {' + f'"name": "{tool_name}", arguments": '
        return prompt