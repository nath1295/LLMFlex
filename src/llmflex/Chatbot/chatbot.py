from typing import TYPE_CHECKING, Optional, List, Iterator, Union, Callable, Dict, Any, Tuple
from pydantic import BaseModel
if TYPE_CHECKING:
    from ..LLM.Engine.base_engine import BaseLLM
    from ..Memory.base_memory import BaseMemory
    from ..Tool.base_tool import BaseTool
    from ..Prompt.chat_template import ChatTemplate

class MemoryConfig(BaseModel):
    recent_token_limit: int = 1000
    relevant_token_limit: int = 500
    relevance_score_threshold: float = 0.0

LONG_TERM_MEMORY_SYSTEM_RPOMPT = """
Bellow are pieces of relevant contents from the older parts of the conversation. If you find the content from these pieces useful, you can use them to help you respond to the user. You do not have to tell the user that you have used the content and just respond naturally.
"""

def direct_response() -> None:
    """If no tool is required to respond to the user, use this tool to respond directly.
    """

class Chatbot:
    """Class providing chatbot functionalities with an LLM.
    """
    def __init__(self,
            llm: "BaseLLM",
            memory: Optional["BaseMemory"] = None,
            tools: Optional[List[Union["BaseTool", Callable]]] = None,
            chat_template: Optional["ChatTemplate"] = None,
        ) -> None:
        """Initializes the chatbot with the provided LLM, memory, tools, and chat template.

        Args:
            llm (BaseLLM): The language model engine to use for generating responses.
            memory (Optional[BaseMemory], optional): The memory system to use for storing and retrieving conversation history. If not provided, a `ChatMemory` instance will be initiated. Defaults to None.
            tools (Optional[List[Union[BaseTool, Callable]]], optional): A list of tools that the chatbot can use to assist with responses. All turned off by default. Defaults to None.
            chat_template (Optional[ChatTemplate], optional): The chat template to use for formatting the conversation. If not provided, the default LLM chat template will be used. Defaults to None.
        """
        from ..Memory.chat_memory import ChatMemory
        from ..Tool.base_tool import FunctionTool, BaseTool
        self._llm = llm
        self._memory = memory if memory else ChatMemory()
        self._tools = [FunctionTool(tool) if not isinstance(tool, BaseTool) else tool for tool in tools] if tools else []
        tool_names = [tool.name for tool in self.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError(f'Multiple tools with the same tool name.')
        self._chat_template = chat_template if chat_template else self._llm.chat_template

    @property
    def llm(self) -> "BaseLLM":
        """LLM for the chatbot.

        Returns:
            BaseLLM: LLM for the chatbot.
        """
        return self._llm
    
    @property
    def memory(self) -> "BaseMemory":
        """Conversation memory.

        Returns:
            BaseMemory: Conversation memory.
        """
        return self._memory
    
    @property
    def tools(self) -> List["BaseTool"]:
        """List of available tools.

        Returns:
            List[BaseTool]: List of available tools.
        """
        return self._tools
    
    @property
    def chat_template(self) -> "ChatTemplate":
        """Chat template for formatting conversation.

        Returns:
            ChatTemplate: Chat template for formatting conversation.
        """
        return self._chat_template
    
    @property
    def title(self) -> Optional[str]:
        """Title of the conversation if set.

        Returns:
            Optional[str]: Title of the conversation if set.
        """
        return self.memory.title
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """Conversation history.

        Returns:
            List[Dict[str, Any]]: Conversation history.
        """
        return self.memory.history
    
    @property
    def tool_dict(self) -> Dict[str, bool]:
        """Dictionary of tools with keys as the tool names and values as whether the tool is turned on or not.

        Returns:
            Dict[str, bool]: Dictionary of tools with keys as the tool names and values as whether the tool is turned on or not.
        """
        if not hasattr(self, '_tool_dict'):
            self._tool_dict = {tool.name: False for tool in self.tools}
        return self._tool_dict
    
    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        """Generation configuration of the LLM.

        Returns:
            Dict[str, Any]: Generation configuration of the LLM.
        """
        if not hasattr(self, '_generation_kwargs'):
            self._generation_kwargs = dict(
                temperature=0.0,
                min_p=0.05,
                repetition_penalty=1.1,
                max_new_tokens=2048,
                stop=[self.llm.tokenizer.eos_token]
            )
        return self._generation_kwargs
    
    @property
    def memory_config(self) -> MemoryConfig:
        """Memory configuration.

        Returns:
            MemoryConfig: Memory configuration.
        """
        if not hasattr(self, '_memory_config'):
            self._memory_config = MemoryConfig()
        return self._memory_config
    
    @property
    def system_message(self) -> Optional[str]:
        """System message of the conversation.

        Returns:
            Optional[str]: System message of the conversation.
        """
        if not hasattr(self, '_system_message'):
            self._system_message = None
        return self._system_message
    
    
    def set_title(self, title: Optional[str]) -> None:
        """Set the title of the conversation.

        Args:
            title (Optional[str]): The new title of the conversation.
        """
        self.memory.set_title(title)

    def set_system_message(self, system_message: str) -> None:
        """Set the system message of the conversation.

        Args:
            system_message (str): The new system message of the conversation.
        """
        if system_message.strip() == '':
            self._system_message = None
        else:
            self._system_message = system_message.strip()
    
    def set_generation_kwargs(self, **kwargs) -> None:
        """Set the generation configuration of the LLM by passing keyword arguments.
        """
        if kwargs:
            self.generation_kwargs.update(kwargs)

    def set_memory_config(self, 
            recent_token_limit: Optional[int] = None,
            relevant_token_limit: Optional[int] = None,
            relevance_score_threshold: Optional[float] = None,
        ) -> None:
        """Set the memory configuration.

        Args:
            recent_token_limit (Optional[int], optional): The maximum number of tokens to retrieve from the latest conversation turns of the memory. Defaults to None.
            relevant_token_limit (Optional[int], optional): The maximum number of tokens to retrieve from the older but relevant conversation turns of the memory.. Defaults to None.
            relevance_score_threshold (Optional[float], optional): The threshold for relevance score. Defaults to None.
        """
        if recent_token_limit is not None:
            self.memory_config.recent_token_limit = recent_token_limit
        if relevant_token_limit is not None:
            self.memory_config.relevant_token_limit = relevant_token_limit
        if relevance_score_threshold is not None:
            self.memory_config.relevance_score_threshold = relevance_score_threshold
        config = self.memory_config.model_dump()
        new_config = dict(
            recent_token_limit=recent_token_limit,
            relevant_token_limit=relevant_token_limit,
            relevance_score_threshold=relevance_score_threshold
        )
        for k, v in new_config.items():
            if v is not None:
                config[k] = v
        self._memory_config = MemoryConfig(**config)

    def toggle_tool(self, tools: Union[str, List[str]]) -> None:
        """Toggle the given tool(s) for their availability of the chatbot.

        Args:
            tools (Union[str, List[str]]): List of tools to toggle.
        """
        ts = [tools] if isinstance(tools, str) else tools
        ts = list(set(ts))
        if any(t not in self.tool_dict.keys() for t in ts):
            raise ValueError('Non-exist tool name given.')
        for t in ts:
            self.tool_dict[t] = not self.tool_dict[t]

    def prepare_prompt(self, user_message: str, tool_call: Optional[Dict[str, Any]] = None) -> str:
        """Prepare the prompt for generation.

            user_message (str): The user's message to be processed.
            tool_call (Optional[Dict[str, Any]], optional): A dictionary containing information about a tool call. Defaults to None.

        Returns:
            str: The prompt to be passed to the LLM.
        """
        system = self.system_message
        if any(v for v in self.tool_dict.values()):
            from ..Tool.tool_utils import create_openai_function
            tools = []
            for k, v in self.tool_dict.items():
                if v:
                    tool = list(filter(lambda x: x.name == k, self. tools))[0]
                    tools.append(create_openai_function(tool))
            tools.append(create_openai_function(direct_response))
        else:
            tools = None
        system = self.system_message.strip() if self.system_message else None
        messages = [dict(role='system', content=system)] if system else []
        recent = self.memory.get_messages_by_token_limit(token_limit=self.memory_config.recent_token_limit, tokenizer=self.llm.tokenizer)
        messages += recent
        if self.memory.__class__.__name__ == 'LongShortTermChatMemory':
            relevant = self.memory.get_related_messages(
                query=user_message,
                token_limit=self.memory_config.relevant_token_limit,
                score_threshold=self.memory_config.relevance_score_threshold,
                exclude_messages=recent
            )
            if len(relevant) > 0:
                import json
                msg = f'<relevant_content>\n{LONG_TERM_MEMORY_SYSTEM_RPOMPT.strip()}\n\n{json.dumps(relevant)}\n</relevant_content>\n\n' + user_message.strip()
            else:
                msg = user_message.strip()
        else:
            msg = user_message.strip()
        messages.append(dict(role='user', content=msg))
        if tool_call:
            assistant_tool_calls = dict(role='assistant', tool_calls=tool_call['tool_calls'])
            tool_output = dict(role='tool', content=tool_call['output'])
            messages.extend([assistant_tool_calls, tool_output])
        prompt = self.chat_template.apply_chat_template(messages=messages, tools=tools)
        return prompt
    
    def chat(self, user_message: str, stream: bool = False) -> Union[str, Tuple[str, Dict[str, Any]], Iterator[Union[Dict[str, Any], str]]]:
        """Generate a response to the user's message.

        Args:
            user_message (str): The user's message to be processed.
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            Union[str, Tuple[str, Dict[str, Any]], Iterator[Union[Dict[str, Any], str]]]: The response to the user's message. A dictionary will be yielded for tool calls.
        """
        prompt = self.prepare_prompt(user_message)
        has_tool = any(v for v in self.tool_dict.values())
        if has_tool:
            tool_prompt = prompt + self.chat_template.tool_start + '{"name": "'
            tool_names = [k for k, v in self.tool_dict.items() if v] + ['direct_response']
            tool_name = self.llm.generate_choice(prompt=tool_prompt, choices=tool_names, **self.generation_kwargs)
            if tool_name == 'direct_response':
                if stream:
                    output = self.llm.stream(prompt=prompt, **self.generation_kwargs)
                    def gen_tokens():
                        response = ''
                        for t in output:
                            response += t
                            yield t
                        self.memory.save_messages(messages=[dict(role='user', content=user_message.strip()), dict(role='assistant', content=response.strip())])
                    return gen_tokens()
                else:
                    output = self.llm.generate(prompt=prompt, **self.generation_kwargs)
                    self.memory.save_messages(messages=[dict(role='user', content=user_message.strip()), dict(role='assistant', content=output.strip())])
                    return output.strip()
            else:
                from ..Tool.tool_utils import create_function_schema
                import json
                tool_prompt += tool_name + '"' + ', "arguments": '
                tool = list(filter(lambda x: x.name == tool_name, self.tools))[0]
                tool_schema = create_function_schema(tool)
                tool_arguments_str = self.llm.generate_json(tool_prompt, json_schema=tool_schema)
                tool_arguments = json.loads(tool_arguments_str)
                if stream:
                    def gen_output():
                        tool_call_dict = dict(function=dict(name=tool_name, arguments=tool_arguments_str))
                        yield tool_call_dict
                        tool_output = tool.run(**tool_arguments)
                        tool_call_dict['output'] = tool_output.model_dump()
                        yield tool_call_dict
                        prompt_dict = dict(tool_calls=[dict(function=tool_call_dict['function'])], output=tool_output.content)
                        tool_prompt = self.prepare_prompt(user_message=user_message, tool_call=prompt_dict)
                        output = ''
                        for t in self.llm.stream(tool_prompt, **self.generation_kwargs):
                            output += t
                            yield t
                        save_msgs = [
                            dict(role='user', content=user_message.strip()),
                            dict(role='assistant', tool_calls=[dict(function=tool_call_dict['function'])]),
                            dict(role='tool', content=tool_output.content, metadata=tool_output.model_dump()),
                            dict(role='assistant', content=output.strip())
                        ]
                        self.memory.save_messages(save_msgs)
                    return gen_output()
                else:
                    tool_call_dict = dict(function=dict(name=tool_name, arguments=tool_arguments_str))
                    tool_output = tool.run(**tool_arguments)
                    tool_call_dict['output'] = tool_output.model_dump()
                    prompt_dict = dict(tool_calls=[dict(function=tool_call_dict['function'])], output=tool_output.content)
                    tool_prompt = self.prepare_prompt(user_message=user_message, tool_call=prompt_dict)
                    output = self.llm.generate(tool_prompt, **self.generation_kwargs)
                    save_msgs = [
                        dict(role='user', content=user_message.strip()),
                        dict(role='assistant', tool_calls=[dict(function=tool_call_dict['function'])]),
                        dict(role='tool', content=tool_output.content, metadata=tool_output.model_dump()),
                        dict(role='assistant', content=output.strip())
                    ]
                    self.memory.save_messages(save_msgs)
                    return output.strip(), tool_call_dict
        
        elif stream:
            output = self.llm.stream(prompt=prompt, **self.generation_kwargs)
            def gen_tokens():
                response = ''
                for t in output:
                    response += t
                    yield t
                self.memory.save_messages(messages=[dict(role='user', content=user_message.strip()), dict(role='assistant', content=response.strip())])
            return gen_tokens()
        else:
            output = self.llm.generate(prompt=prompt, **self.generation_kwargs)
            self.memory.save_messages(messages=[dict(role='user', content=user_message.strip()), dict(role='assistant', content=output.strip())])
            return output.strip()

    def remove_last_interaction(self, keep_last_user: bool = False) -> None:
        """Remove the last interaction from the memory.

        Args:
            keep_last_user (bool, optional): Whether to keep the last user message. Defaults to False.
        """
        reversed_history = self.memory.history[::-1]
        del_next = True
        for msg in reversed_history:
            if del_next:
                if msg['role'] == 'user':
                    del_next = False
                    if keep_last_user:
                        break
                self.memory.remove_last_message()
            else:
                break

    def notebook_chat(self, user_message: str) -> None:
        """Chating with the bot in a Jupyter notebook, with conversation history formatted and displayed as markdown. 

        Args:
            user_message (str): The new user message.
        """
        from IPython.display import display, Markdown, clear_output
        import json
        md = ''
        footnotes = None
        ast_added = False
        for msg in self.history:
            if msg['role'] == 'user':
                md += '**USER:** ' + msg['content'].strip() + '  \n___\n'
            elif msg['role'] == 'assistant':
                if not ast_added:
                    md += '**ASSISTANT:** '
                    ast_added = True
                if msg.get('content', None):
                    md += msg['content'].strip()
                    if footnotes:
                        fn_str = '  \n'.join([f'* {fn}' for fn in footnotes])
                        md += '\n___\n' + fn_str
                        footnotes = None
                    md += '  \n___\n'
                    ast_added = False
                else:
                    tool_name = msg['tool_calls'][0]['function']['name']
                    md += f'  \n<details>\n<summary>{tool_name.replace("_", " ").title()}</summary>\n\n'
            else:
                md += json.dumps(msg['metadata']) + '\n</details>\n  \n'
                images = msg['metadata']['images']
                footnotes = msg['metadata']['footnotes']
                if images:
                    for image in images:
                        md += f'  \n![image]({image})'
                    md += '  \n'

        md += f'**USER:** {user_message.strip()}' + '  \n___\n'
        md += f'**ASSISTANT:** '
        display(Markdown(md))
        output = self.chat(user_message=user_message, stream=True)
        res = ''
        images = None
        footnotes = None
        for i in output:
            if isinstance(i, dict):
                if i.get('output'):
                    expandable = f"  \n<details>\n<summary>{i['function']['name']}</summary>\n\n{json.dumps(i['output'], indent=2)}\n</details>\n  \n"
                    res += expandable
                    images = i['output']['images']
                    footnotes = i['output']['footnotes']
                    clear_output()
                    display(Markdown(md + res))
                else:
                    expandable = f"  \n<details>\n<summary>{i['function']['name']}</summary>\n</details>\n  \n"
                    clear_output()
                    display(Markdown(md + expandable.strip()))
            else:
                if images:
                    for image in images:
                        res += f'  \n![image]({image})'
                    res += ''
                    images = None
                res += i
                clear_output()
                display(Markdown(md + res))
        md += res
        if footnotes:
            fn_str = '  \n'.join([f'* {fn}' for fn in footnotes])
            md += '\n___\n' + fn_str
        clear_output()
        display(Markdown(md))


        


