from ..Prompts.prompt_template import PromptTemplate
from ..Models.Cores.base_core import BaseLLM
from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, Union, Type, Tuple, Any, Dict

class BaseTool(ABC):
    """This is a base class for tools for LLMs.
    """
    def __init__(self, name: str = 'base_tool', description: str = 'This is a tool from the base tool class. It does not do anything.', verbose: bool = True) -> None:
        """Initialising the tool.
        """
        self._name = name
        self._description = description
        self._verbose = verbose

    @property
    def name(self) -> str:
        """Name of the tool.

        Returns:
            str: Name of the tool.
        """
        return self._name
    
    @property
    def pretty_name(self) -> str:
        """Pretty name of the tool.

        Returns:
            str: Pretty name of the tool.
        """
        pretty = self.name.replace('_', ' ')
        return pretty.title()
    
    @property
    def description(self) -> str:
        """Description of the tool.

        Returns:
            str: Description of the tool.
        """
        import re
        newlines = re.compile(r'[\s\r\n\t]+')
        return newlines.sub(' ', self._description)
    
    @property
    def _avail_steps(self) -> List[str]:
        """List of available steps of the tool.

        Returns:
            List[str]: List of available steps of the tool.
        """
        import inspect
        attrs = dir(self)
        core_methods = ['run', 'print', '_execute_step', '_tool_schema', '_validate_schema', '_avail_steps']
        steps = list(filter(lambda x: x not in core_methods, attrs))
        steps = list(filter(lambda x: inspect.ismethod(getattr(self, x)), steps))
        return steps
    
    def print(self, text: str, **kwargs) -> None:
        """Print the given text if verbose is True.

        Args:
            text (str): Text to print.
        """
        if self._verbose:
            print(text, **kwargs)

    def _execute_step(self, step: str, **kwargs) -> Any:
        """Executing a step (a method in your tool class).

        Args:
            step (str): Step to execute.

        Returns:
            Any: Output of the step.
        """
        if step not in self._avail_steps:
            raise ValueError(f'Step "{step}" not implemented yet.')
        method = getattr(self, step)
        return method(**kwargs)
    
    @abstractmethod
    def _tool_schema(self) -> Dict[str, Any]:
        """Defining how the run method runs the tool. This method is the most important method of the tool class and must be implemented properly.
        The returned dictionary is the order of running the tool steps, with the keys as the name of the steps and the value as a dictionary input and output of the step.
        Example: {
            "step1": {"input": ["tool_input"], "output": ["var1", "var2"]}, 
            "step2": {"input": ["var1"], "output": ["var3"]}, 
            "final_step": {"input": ["var2", "var3", "llm", "stream", "prompt_template"], "output": ["final_output"]}
            }
        As demonstrated in the example, the final step must have the "stream" and "prompt_template" arguments to allow the tool to return a string or an iterator of string as the llm response.

        Returns:
            Dict[str, Any]: Tool run schema dictionary.
        """
        pass

    def _validate_schema(self, **kwargs) -> None:
        """Raise an error if the _tool_schema is not implemented properly.
        """
        import inspect
        schema = self._tool_schema()
        steps = list(schema.keys())
        if len(steps) == 0:
            raise RuntimeError(f'"_tool_schema" cannot return an empty dictinoary.')
        final_step = list(schema.keys())[-1]
        var_space = ["tool_input", "llm", "stream", "history", "prompt_template"] + list(kwargs.keys())
        for k, v in schema.items():
            if type(v) != dict:
                raise TypeError(f'Value for step "{k}" is not a dictionary.')
            for i in ['input', 'output']:
                if i not in v.keys():
                    raise ValueError(f'"{i}" not in the dictionary of step "{k}".')
                if type(v[i]) != list:
                    raise ValueError(f'"{i}" of step "{k}" is not a list.')
            if k == final_step:
                if "stream" not in v['input']:
                    raise ValueError(f'Argument "stream" not in the input of the final step "{k}".')
                if "llm" not in v['input']:
                    raise ValueError(f'Argument "llm" not in the input of the final step "{k}".')
                if "final_output" not in v['output']:
                    raise ValueError(f'Argument "final_output" not in the output of the final step "{k}".')
                if inspect.signature(getattr(self, k)).return_annotation != Union[str, Iterator[str]]:
                    raise TypeError(f'The output of the final step "{k}" might not be "Union[str, Iterator[str]]", if it is please add the appropriate type hinting.')
            if any(i not in var_space for i in v['input']):
                raise ValueError(f'Some inputs of step "{k}" are not created prior to executing the step.')
            var_space.extend(v['output'])

    def run(self, tool_input: str, llm: Type[BaseLLM], prompt_template: Optional[PromptTemplate] = None, 
            stream: bool = False, history: Optional[Union[List[str], List[Tuple[str, str]]]] = None, add_footnote: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Run the tool and return the output as a string or an iterator of strings.

        Args:
            tool_input (str): String input for to run the tool.
            llm (Type[BaseLLM]): LLM to generate the output in a conversational setup.
            prompt_template (Optional[PromptTemplate], optional): prompt_template to format the chat history and create final output. If not given, the llm default prompt template will be used. Defaults to None.
            stream (bool, optional): Whether to stream the output, if True, a generator of the output will be returned. Defaults to False.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): Snippet of chat history to help running the tool if required. Defaults to None.
            add_footnote (bool, optional): Whether to append to footnote to the output. Defaults to False.

        Returns:
            Union[str, Iterator[str]]: Output of the tool.
        """
        self._validate_schema(**kwargs)
        var_space = dict(
            tool_input = tool_input,
            llm = llm,
            prompt_template = llm.core.prompt_template if prompt_template is None else prompt_template,
            stream = stream,
            history = history
        )
        var_space.update(kwargs)
        for k, v in self._tool_schema().items():
            input_dict = {i: var_space[i] for i in v['input']}
            output_dict = dict()
            self.print(k)
            if len(v['output']) == 0:
                self._execute_step(k, **input_dict)
            elif len(v['output']) == 1:
                output_dict[v['output'][0]] = self._execute_step(k, **input_dict)
            else:
                outputs = self._execute_step(k, **input_dict)
                output_dict = dict(zip(v['output'], outputs))
            self.print('\tOutputs:')
            for s, o in output_dict.items():
                if s != 'final_output':
                    limit = 100
                    output = str(o).replace('\n', ' ')
                    output = output.replace('\r', ' ')
                    output = output if len(output) <= limit else output[:limit] + '...'
                    self.print(f'\t{s}: {output}')
            self.print('\n')
            var_space.update(output_dict)
        final = var_space['final_output']
        if (('footnote' in var_space.keys()) & add_footnote):
            final += f'\n\n---\n{var_space["footnote"]}'
        return final

    
    def run_with_chat(self, tool_input: str, llm: Type[BaseLLM], prompt_template: Optional[PromptTemplate] = None, 
            stream: bool = False, history: Optional[Union[List[str], List[Tuple[str, str]]]] = None, 
            add_footnote: bool = True, **kwargs) -> Iterator[Union[str, Tuple[str, str], Iterator[str]]]:
        """Running tool with chat, it will yield the markdown friendly string of tool info for each steps and the final output, along with any extra information after the final output.

        Args:
            tool_input (str): String input for to run the tool.
            llm (Type[BaseLLM]): LLM to generate the output in a conversational setup.
            prompt_template (Optional[PromptTemplate], optional): prompt_template to format the chat history and create final output. If not given, the llm default prompt template will be used. Defaults to None.
            stream (bool, optional): Whether to stream the output, if True, a generator of the output will be returned. Defaults to False.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): Snippet of chat history to help running the tool if required. Defaults to None.
            add_footnote (bool, optional): Whether to append to footnote to the output. Defaults to True.

        Yields:
            Iterator[Union[str, Iterator[str]]]: Iterator of the markdown friendly string of tool info for each steps and the final output, along with any extra information after the final output.
        """
        self._validate_schema(**kwargs)
        var_space = dict(
            tool_input = tool_input,
            llm = llm,
            prompt_template = llm.core.prompt_template if prompt_template is None else prompt_template,
            stream = stream,
            history = history
        )
        var_space.update(kwargs)
        info = []
        name = self.name.replace('_', ' ').title()
        header = f'Running "{name}"...'
        yield (header, '\n'.join(info))

        for k, v in self._tool_schema().items():
            input_dict = {i: var_space[i] for i in v['input']}
            output_dict = dict()
            info.append(f'{k}:  ')
            yield (header + k, '\n'.join(info))
            if len(v['output']) == 0:
                self._execute_step(k, **input_dict)
            elif len(v['output']) == 1:
                output_dict[v['output'][0]] = self._execute_step(k, **input_dict)
            else:
                outputs = self._execute_step(k, **input_dict)
                output_dict = dict(zip(v['output'], outputs))
            info.append('\tOutputs:  ')
            for s, o in output_dict.items():
                if s != 'final_output':
                    limit = 100
                    output = str(o).replace('\n', ' ')
                    output = output.replace('\r', ' ')
                    output = output if len(output) <= limit else output[:limit] + '...'
                    info.append(f'\t{s}: {output}  ')
            info.append('\n')
            var_space.update(output_dict)
            yield (header + k, '\n'.join(info))
        yield (f'"{name}" completed.', '\n'.join(info))
        yield var_space['final_output']

        if (('footnote' in var_space.keys()) & add_footnote):
            yield var_space["footnote"]

        
