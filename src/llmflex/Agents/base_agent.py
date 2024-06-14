from ..Tools.tool_utils import ToolSelector, BaseTool
from ..Prompts.prompt_template import PromptTemplate
from ..Models.Cores.base_core import BaseLLM
from typing import Type, List, Optional, Union, Callable, Iterator, Dict, Literal, Any

GENERIC_EXAMPLES = """Task given by user: How old will Joe Biden be after three years?

"thought": "I need to know the current age of Joe Biden first."
"action_input": {
    "name": "browser_tool",
    "inputs": {
        "search_query": "Joe Biden current age"
    }
}
"action_output": ... (Some contents that mentioned Joe Biden's age, let's say 81 years old)
"thought": "Now I know that the age of Joe Biden is 81, I will need to calculate his age after three years."
"action_input": {
    "name": "math_tool",
    "inputs": {
        "equation": "81 + 3"
    } 
}
"action_output": 84
"thought": "I think I know the answer now."
"action_input": {
    "name": "direct_response"
}
"response": "Joe Biden will be 84 years old after three years."
"""

GENERIC_AGENT_SYSTEM_MESSAGE = """You are an intelligent agent with access to the a range of functions. Solve the task given by the user by going through a series of thought-action loop with the functions you have access to. Think step by step and only use the "direct_response" function when you have a solution to the task given.

Here is an example of How you should work.
```
{examples}
```

You will be given the information of the functions you have access to and the task given by the user. Solve the task with the given format in the example.
"""

class AgentProcess:
    """Class for hosting agent step outputs.
    """
    def __init__(self, role: Literal['thought', 'action_input', 'action_output', 'response'], content: Any) -> None:
        self._role = role
        self._content = content

    @property
    def role(self) -> Literal['thought', 'action_input', 'action_output', 'response']:
        return self._role
    
    @property
    def content(self) -> Any:
        return self._content
    
    def __repr__(self) -> str:
        return f'AgentProcess(role={self.role}, content={self.content})'
    
class Agent:
    """Base agent class.
    """
    def __init__(self, llm: Type[BaseLLM], 
                tools: Optional[List[Union[Callable, Type[BaseTool]]]] = None, 
                tool_selector: Optional[ToolSelector] = None, 
                prompt_template: Optional[PromptTemplate] = None,
                system_message: Optional[str] = None,
                examples: Optional[str] = None
            ) -> None:
        """Initialise the agent.

        Args:
            llm (Type[BaseLLM]): The llm of the agent.
            tools (Optional[List[Union[Callable, Type[BaseTool]]]], optional): List of tools for the agent, can be None if the tool selector is provided. Defaults to None.
            tool_selector (Optional[ToolSelector], optional): Tool selector for the agent, can be None if a list of tools is given. Defaults to None.
            prompt_template (Optional[PromptTemplate], optional): Prompt template for the agent, If None, the default prompt template of the llm will be used. Defaults to None.
            system_message (Optional[str], optional): System message for the agent. If None, the GENERIC_AGENT_SYSTEM_MESSAGE will be used. Defaults to None.
            examples (Optional[str], optional): Examples for the agent. If None, the GENERIC_EXAMPLES will be used. Defaults to None.
        """
        self._llm = llm
        self._prompt_template = self.llm.core.prompt_template if prompt_template is None else prompt_template
        if tools is None and tool_selector is None:
            raise RuntimeError('One of tool selector or a list of tools must be provided.')
        self._tool_selector = ToolSelector(tools=tools) if tool_selector is None else tool_selector
        system_format = GENERIC_AGENT_SYSTEM_MESSAGE if system_message is None else system_message
        examples = GENERIC_EXAMPLES if examples is None else examples
        self._system = system_format.format(examples=examples)
        self._verbose = True
        self._log = []
    
    @property
    def llm(self) -> BaseLLM:
        """The llm of the agent.

        Returns:
            BaseLLM: The llm of the agent.
        """
        return self._llm

    @property
    def prompt_template(self) -> PromptTemplate:
        """Prompt template used by the agent.

        Returns:
            PromptTemplate: Pormpt template used by the agent.
        """
        return self._prompt_template
    
    @property
    def tool_selector(self) -> ToolSelector:
        """Tool selector of the agent.

        Returns:
            ToolSelector: Tool selector of the agent.
        """
        return self._tool_selector
    
    @property
    def system(self) -> str:
        """System message of the agent.

        Returns:
            str: System message of the agent.
        """
        return self._system
    
    @property
    def log(self) -> List[AgentProcess]:
        """Logs of the latest run of the agent.

        Returns:
            List[AgentProcess]: Logs of the latest run of the agent.
        """
        return self._log

    def show(self, content: Any, role: Optional[str] = None, max_char: Optional[int] = None, end: str = '\n') -> None:
        """Print the given content.

        Args:
            content (Any): Content to print.
            role (Optional[str], optional): Prefix for the content. Defaults to None.
            max_char (Optional[int], optional): Maximum character of the content to print. Defaults to None.
            end (str, optional): End character for the print function. Defaults to '\n'.
        """
        if self._verbose:
            import json
            if isinstance(content, str):
                print_str = content
            else:
                try:
                    print_str = json.dumps(content)
                except:
                    print_str = str(content)
            if max_char is not None:
                if len(print_str) >= max_char:
                    print_str = print_str[:max_char] + '...'
            if role is not None:
                print_str = f'{role}: {print_str}'
            print(print_str, end=end)

    def prerun_config(self, **kwargs) -> None:
        """Any processes that should be run before the main action reaction loop.
        """
        pass

    def final_response(self, prompt: str, response_stream: bool  = False) -> AgentProcess:
        """Handling final direct response from the agent.

        Args:
            prompt (str): The full prompt for direct response.
            response_stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            AgentProcess: Agent process with the final response (Or response stream).
        """
        from ..Tools.tool_utils import gen_string
        if response_stream:
            process = AgentProcess(role='response', content=self.llm.stream(prompt + '"', stop=self.prompt_template.stop + ['"']))
            return process
        else:
            self.show(content='Response: ', end='')
            response = gen_string(self.llm, prompt=prompt).strip()
            self.show(response)
            process = AgentProcess(role='response', content=response)
            self._log.append(process)
            return process

    def run_iter(self, task: str, max_iter: int = 10, verbose: bool = True, response_stream: bool = False, force_response: bool = True, **kwargs) -> Iterator[AgentProcess]:
        """Run the agent and yield all itermediate processes.

        Args:
            task (str): Task for the agent.
            max_iter (int, optional): Maximum number of thought-action iteractions. Defaults to 10.
            verbose (bool, optional): Whether to print the processes. Defaults to True.
            response_stream (bool, optional): Whether to return the final response as an iterator of text tokens. Defaults to False.
            force_response (bool, optional): When the max_iter is hit, whether to force a response. Defaults to True.

        Yields:
            Iterator[AgentProcess]: Iterator a agent processes.
        """
        import json
        from ..Tools.tool_utils import gen_string
        self._log = []
        self._verbose = verbose
        self.prerun_config(**kwargs)
        if self.prompt_template.allow_custom_role:
            messages = [
                dict(role='system', content=self.system),
                self.tool_selector.function_metadata,
                dict(role='user', content=task.strip())
            ]
        else:
            system = self.system + '\n\n' + json.dumps(dict(function_metadata=self.tool_selector.function_metadata['content']), indent=4)
            messages = [
                dict(role='system', content=system),
                dict(role='user', content=task.strip())
            ]
        prompt = self.prompt_template.create_custom_prompt(messages=messages)
        for i in range(max_iter):
            # Generating thought
            prompt += '"thought": '
            self.show(content='Thought: ', end='')
            thought = gen_string(self.llm, prompt=prompt).strip()
            prompt += f'"{thought}"'
            self.show(thought)
            process = AgentProcess(role='thought', content=thought)
            self._log.append(process)
            yield process

            # Generating action input
            prompt += '\n"action_input": '
            self.show(content='Action Input: ', end='')
            action_input = self.tool_selector.structured_input_generation(prompt, llm=self.llm)
            prompt += json.dumps(action_input, indent=4)
            self.show(action_input)
            process = AgentProcess(role='action_input', content=action_input)
            self._log.append(process)
            yield process

            # Generate response if action input is "direct_response"
            if action_input['name'] == 'direct_response':
                prompt += '\n"response": '
                yield self.final_response(prompt, response_stream=response_stream)
            
            # Generate action output
            else:
                prompt += '\n"action_output": '
                self.show(content='Action Output: ', end='')
                action_output = self.tool_selector.tool_call_output(action_input, return_error=True)
                if 'error' in action_output.keys():
                    action_output = 'error - ' + action_output['error']
                else:
                    action_output = action_output['output']
                try:
                    output_str = json.dumps(action_output, indent=4)
                except:
                    output_str = f'"{str(action_output)}"'
                prompt += output_str + '\n'
                self.show(action_output, max_char=100)
                process = AgentProcess(role='action_output', content=action_output)
                self._log.append(process)
                if i != max_iter - 1:
                    self.show('')
                yield process
            
        if force_response:
            prompt += '"response": '
            yield self.final_response(prompt, response_stream=response_stream)

    def run(self, task: str, max_iter: int = 10, verbose: bool = True, response_stream: bool = False, force_response: bool = True, **kwargs) -> Union[str, Iterator[str]]:
        """Run the agent.

        Args:
            task (str): Task for the agent.
            max_iter (int, optional): Maximum number of thought-action iteractions. Defaults to 10.
            verbose (bool, optional): Whether to print the processes. Defaults to True.
            response_stream (bool, optional): Whether to return the final response as an iterator of text tokens. Defaults to False.
            force_response (bool, optional): When the max_iter is hit, whether to force a response. Defaults to True.

        Returns:
            Union[str, Iterator[str]]: Final response.
        """
        step_generator = self.run_iter(
            task=task,
            max_iter=max_iter,
            verbose=verbose,
            response_stream=response_stream,
            force_response=force_response,
            **kwargs
        )
        for step in step_generator:
            if step.role == 'response':
                return step.content
            
        return 'Reached max iteration.'

