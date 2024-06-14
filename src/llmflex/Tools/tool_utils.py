import inspect
from typing import Literal, List, Dict, Any, Optional, Callable, Union, Type
from ..Models.Cores.base_core import BaseLLM
from ..Prompts.prompt_template import PromptTemplate

PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array"
}

def normalise_tool_name(name: str) -> str:
    """Make the tool name in lower case and split it with "_".

    Args:
        name (str): Original name of the tool.

    Returns:
        str: Normalised tool name.
    """
    import re
    tokens = re.sub( r"([A-Z])", r" \1", name).split()
    tokens = list(map(lambda x: x.lower().strip(' \n\r\t_'), tokens))
    return '_'.join(tokens).strip()

def get_description(docstring: str) -> str:
    """Get the function or tool description from the docstring.

    Args:
        docstring (str): The docstring of the function or tool.

    Returns:
        str: The description of the function or tool.
    """
    blocks = ['args:', 'raises:', 'returns:']
    lines = map(lambda x: x.strip(), docstring.split('\n'))
    lines = filter(lambda x: x != '', lines)
    descriptions = []
    for line in lines:
        if line.lower() in blocks:
            break
        else:
            descriptions.append(line)
    return '\n'.join(descriptions).strip()

class BaseTool:
    """Base class for tools.
    """
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None) -> None:
        """Initialising the tool.

        Args:
            name (Optional[str], optional): Name of the tool. If not given, it will be the tool class name. Defaults to None.
            description (Optional[str], optional): Description of the tool. If not given, it will read from the docstring of the tool class. Defaults to None.
        """
        from ..utils import validate_type
        self._name = normalise_tool_name(validate_type(name.strip(), str)) if name is not None else normalise_tool_name(self.__class__.__name__)
        self._description = validate_type(description.strip(), str) if description is not None else get_description(self.__doc__) if self.__doc__ else ''

    @property
    def name(self) -> str:
        """Name of the tool.

        Returns:
            str: Name of the tool.
        """
        return self._name
    
    @property
    def description(self) -> str:
        """Description of the tool.

        Returns:
            str: Description of the tool.
        """
        return self._description
    
    def __call__(self, **kwargs) -> Any:
        """The main function of the tool.

        Returns:
            Any: Output of the tool.
        """
        pass

def get_args_descriptions(docstring: str) -> Dict[str, str]:
    """Get the description of the arguments of a function or tool given the docstring.

    Args:
        docstring (str): Docstring of the callable.

    Returns:
        Dict[str, str]: Dictionary of the arguments and their descriptions.
    """
    blocks = ['raises:', 'returns:']
    lines = map(lambda x: x.strip(), docstring.split('\n'))
    lines = filter(lambda x: x != '', lines)
    started = False
    args = dict()
    for line in lines:
        if line.lower() == 'args:':
            started = True
        elif line.lower() in blocks:
            started = False
        elif started:
            elements = line.split(':')
            if len(elements) > 1:
                args[elements[0].split('(')[0].strip()] = ':'.join(elements[1:]).strip()
    return args
                
def get_required_args(fn: Callable) -> List[str]:
    """Get the mandatory arguments of the given function.

    Args:
        fn (Callable): Function of interest.

    Returns:
        List[str]: List of mandatory arguments.
    """
    specs = inspect.getfullargspec(fn)
    args = specs.args
    if specs.defaults:
        args = args[:-len(specs.defaults)]
    args = list(filter(lambda x: x not in ['cls', 'self'], args))
    return args

def get_args_dtypes(fn: Callable) -> Dict[str, Dict[str, Any]]:
    """Get the data types of the arguments of a function or tool given the callable.

    Args:
        fn (Callable): The function or the tool.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of the arguments and their data types and descriptions.
    """
    annotations = inspect.getfullargspec(fn).annotations
    args_desc = get_args_descriptions(fn.__doc__) if fn.__doc__ is not None else dict()
    required = get_required_args(fn)
    signature = inspect.signature(fn)
    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    args = dict()
    for arg, arg_type in annotations.items():
        if arg in ['return', 'cls', 'self']:
            continue

        if getattr(arg_type, '__name__').lower() in PYTHON_TO_JSON_TYPES.keys():
            args[arg] = dict(type=PYTHON_TO_JSON_TYPES[arg_type.__name__.lower()])
            if arg in args_desc:
                args[arg]['description'] = args_desc[arg]
            if arg not in required:
                args[arg]['default'] = defaults[arg]
        elif getattr(arg_type, '__name__') == 'Literal':
            dtype = getattr(type(arg_type.__args__[0]), '__name__')
            dtype = PYTHON_TO_JSON_TYPES.get(dtype, None)
            args[arg] = dict()
            if dtype is not None:
                args[arg]['type'] = dtype
                if arg in args_desc:
                    args[arg]['description'] = args_desc[arg]
                args[arg]['enum'] = list(arg_type.__args__)
                if arg not in required:
                    args[arg]['default'] = defaults[arg]
            else:
                args[arg]['type'] = getattr(arg_type, '__name__', str(arg_type))
                if arg in args_desc:
                    args[arg]['description'] = args_desc[arg]
                args[arg]['enum'] = list(map(lambda x: str(x), arg_type.__args__))
                if arg not in required:
                    args[arg]['default'] = defaults[arg]
        elif getattr(arg_type, '__name__') == 'Optional':
            dtype = getattr(arg_type.__args__[0], '__name__')
            dtype = PYTHON_TO_JSON_TYPES.get(dtype, None)
            args[arg] = dict()
            if dtype is not None:
                args[arg]['type'] = dtype
            else:
                args[arg]['type'] = getattr(arg_type, '__name__', str(arg_type))
            if arg in args_desc:
                args[arg]['description'] = args_desc[arg]
            if arg not in required:
                args[arg]['default'] = defaults[arg]
        else:
            args[arg] = dict()
            args[arg]['type'] = getattr(arg_type, '__name__', str(arg_type))
            if arg in args_desc:
                args[arg]['description'] = args_desc[arg]
            if arg not in required:
                args[arg]['default'] = defaults[arg]
    return args

def get_tool_metadata(fn: Union[Callable, Type[BaseTool]]) -> Dict[str, Any]:
    """Get the tool metadata as a dictionary.

    Args:
        fn (Union[Callable, Type[BaseTool]]): Tool or python function.

    Returns:
        Dict[str, Any]: Metadata of the tool or function.
    """
    is_tool = isinstance(fn, BaseTool)
    desc_doc = fn.description if isinstance(fn, BaseTool) else getattr(fn, '__doc__')
    if is_tool:
        name = fn.name
        description = desc_doc
        required = get_required_args(fn.__call__)
        args = get_args_dtypes(fn.__call__)
    else:
        name = normalise_tool_name(fn.__name__)
        description = get_description(desc_doc) if desc_doc else ''
        required = get_required_args(fn)
        args = get_args_dtypes(fn)
    return dict(name=name, description=description, parameters=dict(type='object', properties=args, required=required))
    
def select(llm: Type[BaseLLM], prompt: str, options: List[str], 
           stop: Optional[List[str]] = None, default: Optional[str] = None, retry: int = 3, raise_error: bool = False, **kwargs) -> Optional[str]:
    """Ask the LLM to make a selection of the list of options provided.

    Args:
        llm (Type[BaseLLM]): LLM to generate the selection.
        prompt (str): Prompt for the llm.
        options (List[str]): List of options for the LLM to pick.
        stop (Optional[List[str]], None): List of stop words for the LLM to help the llm stop earlier. Defaults to None.
        default (Optional[str], optional): Default value if the LLM fails to generate the option. If none is given and the LLM fail, an error will be raised. Defaults to None.
        retry (int, optional): Number of times the llm can retry. Defaults to 3.
        raise_error (bool, optional): Whether to raise an error if the selection failed. Defaults to False.

    Returns:
        Optional[str]: The selection of the LLM given the options.
    """
    if len(options) == 0:
        raise ValueError('"options" must contain at least one string.')
    if llm.core.core_type =='llama_cpp_core':
        from guidance.models.llama_cpp import LlamaCpp
        from guidance import select as gselect
        gllm = LlamaCpp(model=llm.core.model, echo=False)
        res = gllm + prompt + gselect(options=options)
        res = str(res).removeprefix(prompt)
        del gllm
        return res
    else:
        from math import ceil
        from copy import deepcopy
        option_num_tokens = [llm.get_num_tokens(opt) for opt in options]
        max_new_tokens = ceil(max(option_num_tokens) * 1.2)
        stop = [] if stop is None else stop
        for i in range(retry):
            clone_kw = deepcopy(kwargs)
            clone_kw.pop('max_new_tokens', None)
            res = llm.invoke(prompt, stop=stop, max_new_tokens=max_new_tokens, **clone_kw)
            if res in options:
                return res
            else:
                valids = list(filter(lambda x: res.startswith(x), options))
                if len(valids) >= 1:
                    return max(valids, key=len)
        if default is not None:
            return default
        elif raise_error:
            raise RuntimeError(f'LLM cannot generate one of the options after {retry} retries.')
        else:
            return None
        
def gen_string(llm: Type[BaseLLM], prompt: str, double_quote: bool = True, max_gen: int = 5, **kwargs) -> str:
    """Generate a valid string that can be wrapped between quotes safely. 

    Args:
        llm (Type[BaseLLM]): LLM for the string generation.
        prompt (str): Prompt for generating the string, should not include the open quote.
        double_quote (bool, optional): Whether to use double quote or not. The quote character will be added to the end of the original prompt. Defaults to True.
        max_gen (int, optional): In the unlikely event of the LLM keep generating without being able to generate a valid string, this set the maximum of generation the LLM can go. Defaults to 5.

    Returns:
        str: A valid string that can be wrapped between quotes safely. If a valid string cannot be generated, an empty string will be returned.
    """
    quote = '"' if double_quote else "'"
    original = prompt + quote
    text = quote
    not_string: bool = True
    count = 0
    while ((not_string) & (count < max_gen)):
        try:
            assert isinstance(eval(text), str)
            not_string = False
        except:
            try:
                # try to fix issue with newline characters
                original = original.removesuffix(text)
                text = text.replace('\n', '\\n')
                original += text
                assert isinstance(eval(text), str)
                not_string = False
            except:
                new = llm.invoke(original, stop=[quote], **kwargs)
                text += new + quote
                original += new + quote
                count += 1
    if not not_string:
        return eval(text)
    else:
        return ''

def direct_response() -> str:
    """Direct response without using any tools or functions.

    Returns:
        str: Response.
    """
    # This is a placeholder function for function call when no tools or functions are required. 
    pass

### Tool selection
class ToolSelector:
    """Class for selecting tool.
    """
    def __init__(self, tools: List[Union[Callable, Type[BaseTool]]]) -> None:
        self._tools = tools if direct_response in tools else tools + [direct_response]
        self._metadatas = [get_tool_metadata(tool) for tool in self.tools]
        self._tool_names: List[str] = [m['name'] for m in self._metadatas]
        self._enabled = dict(zip(self._tool_names, [True] * len(self._tool_names)))
    
    @property
    def tool_names(self) -> List[str]:
        """List of tool names.

        Returns:
            List[str]: List of tool names.
        """
        return self._tool_names
    
    @property
    def enabled_tools(self) -> List[str]:
        """List of tool names that are enabled.

        Returns:
            List[str]: List of tool names that are enabled.
        """
        enabled_tools = filter(lambda x: x[1], self._enabled.items())
        return list(map(lambda x: x[0], enabled_tools))
    
    @property
    def num_tools(self) -> int:
        """Number of enabled tools.

        Returns:
            int: Number of enabled tools.
        """
        return len(self.enabled_tools)
    
    @property
    def tools(self) -> List[Union[Callable, Type[BaseTool]]]:
        """List of available tools.

        Returns:
            List[Union[Callable, Type[BaseTool]]]: List of available tools.
        """
        return self._tools

    @property
    def metadatas(self) -> List[Dict[str, Any]]:
        """List of metadatas of tools.

        Returns:
            List[Dict[str, Any]]: List of metadatas of tools.
        """
        return list(filter(lambda x: x['name'] in self.enabled_tools, self._metadatas))
    
    @property
    def tool_map(self) -> Dict[str, Union[Callable, Type[BaseTool]]]:
        """Map of tool names and the tools.

        Returns:
            Dict[str, Union[Callable, Type[BaseTool]]]: Map of tool names and the tools.
        """
        if not hasattr(self, '_tool_map'):
            names = [m['name'] for m in self._metadatas]
            self._tool_map = dict(zip(names, self.tools))
        return {k: self._tool_map[k] for k in self.enabled_tools}
    
    @property
    def function_metadata(self) -> Dict[str, str]:
        """A message with the role "function_metadata" and content being the metadata of the tools.

        Returns:
            Dict[str, str]: A message with the role "function_metadata" and content being the metadata of the tools.
        """
        import json
        return dict(role='function_metadata', content=json.dumps(self.metadatas, indent=4))
        
    @property
    def is_empty(self) -> bool:
        """Whether tools except direct_response are enabled.

        Returns:
            bool: Whether tools except direct_response are enabled.
        """
        return ((len(self.enabled_tools) == 1) & ('direct_response' in self.enabled_tools))

    def get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """Get the tool metadata given the tool name.

        Args:
            tool_name (str): Tool name.

        Returns:
            Dict[str, Any]: Metadata of the tool.
        """
        return list(filter(lambda x: x['name'] == tool_name, self.metadatas))[0]
    
    def structured_input_generation(self, raw_prompt: str, llm: Type[BaseLLM], return_raw: bool = False, **kwargs) -> Dict[str, Any]:
        """Core part of tool input generation.

        Args:
            raw_prompt (str): The starting prompt.
            llm (Type[BaseLLM]): LLM for generation.
            return_raw (bool, optional): Whether to return the raw string of failed generation. If False, "direct_response" tool will be returned. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing the name of the function and the input arguments.
        """
        import json
        original_prompt = raw_prompt
        prompt: str = original_prompt + '{\n\t"name": "'
        tool_name = select(llm=llm, prompt=prompt, options=self.enabled_tools, default='direct_response', **kwargs)
        tool = self.get_tool_metadata(tool_name=tool_name)

        # gather tool info
        args_info = tool['parameters']['properties']
        args = list(args_info.keys())
        required_args = tool['parameters']['required']
        optional_args = list(filter(lambda x: x not in required_args, args))

        prompt += tool_name + '"'
        if len(args) == 0:
            prompt += '\n}'
        else:
            prompt += ',\n\t"inputs": {'
            if len(required_args) != 0:
                for arg in required_args:
                    prompt += f'\n\t\t"{arg}": '
                    enum = args_info[arg].get('enum')
                    if args_info[arg].get('type') == 'string':
                        if enum is None:
                            val = gen_string(llm=llm, prompt=prompt, **kwargs)
                            prompt += '"' + val + '",'
                        else:
                            val = select(llm, prompt=prompt + '"', options=enum, **kwargs)
                            if val is not None:
                                prompt +=  f'"{val}"' + ','
                            else:
                                if return_raw:
                                    return dict(generated_text = prompt.removeprefix(original_prompt))
                                else:
                                    return dict(name='direct_response')
                    elif args_info[arg].get('type') == 'array':
                        val = llm.invoke(prompt + '[', stop=[']'], **kwargs)
                        try:
                            is_list = isinstance(eval(f'[{val}]'), list)
                            if is_list:
                                prompt += f'[{val}],'
                            else:
                                if return_raw:
                                    return dict(generated_text = (prompt + f'[{val}]').removeprefix(original_prompt))
                                else:
                                    return dict(name='direct_response')
                        except:
                            if return_raw:
                                return dict(generated_text = (prompt + f'[{val}]').removeprefix(original_prompt))
                            else:
                                return dict(name='direct_response')
                    elif enum is None:
                        prompt += llm.invoke(prompt, stop=['\n\t'], **kwargs)
                    else:
                        val = select(llm, prompt=prompt + '"', options=enum, **kwargs)
                        if val is not None:
                            prompt += val + ','
                        else:
                            if return_raw:
                                return dict(generated_text = prompt.removeprefix(original_prompt))
                            else:
                                return dict(name='direct_response')
                prompt = prompt.rstrip(',') # strip off the comma, see if the llm wants to continue on optional arguments
                if len(optional_args) > 0:
                    from copy import deepcopy
                    clone_kw = deepcopy(kwargs)
                    clone_kw.pop('max_new_tokens', None)
                    new_tokens = llm.invoke(prompt, max_new_tokens=3, **clone_kw)
                    used_args = []
                    while ((new_tokens.startswith(',')) & (len(used_args) < len(optional_args))):
                        prompt += ',\n\t\t"'
                        arg = select(llm=llm, prompt=prompt, options=list(filter(lambda x: x not in used_args, optional_args)), stop=['"'], **kwargs)
                        if arg is None:
                            prompt = prompt.removesuffix(',\n\t\t"')
                            new_tokens = ''
                        else:
                            prompt += arg + '": '
                            enum = args_info[arg].get('enum')
                            default = args_info[arg].get('default')
                            default = 'null' if default is None else default
                            if args_info[arg].get('type') == 'string':
                                if enum is None:
                                    val = gen_string(llm=llm, prompt=prompt, **kwargs)
                                    val = default if val == '' else f'"{val}"'
                                    prompt += val
                                else:
                                    val = select(llm, prompt=prompt + '"', options=enum, **kwargs)
                                    val = default if val is None else f'"{val}"'
                                    prompt += val
                            elif args_info[arg].get('type') == 'array':
                                val = llm.invoke(prompt + '[', stop=[']'], **kwargs)
                                try:
                                    is_list = isinstance(eval(f'[{val}]'), list)
                                    if is_list:
                                        prompt += f'[{val}],'
                                    else:
                                        prompt += f'{default},'
                                except:
                                    prompt += f'{default},'
                            elif enum is None:
                                prompt += llm.invoke(prompt, stop=['\n\t'], **kwargs)
                                prompt = prompt.strip(', \n\r\t')
                            else:
                                val = select(llm, prompt=prompt, options=enum, **kwargs)
                                val = default if val is None else val
                                prompt += val
                            clone_kw = deepcopy(kwargs)
                            clone_kw.pop('max_new_tokens', None)
                            new_tokens = llm.invoke(prompt, max_new_tokens=3, **clone_kw)
            prompt += '\n\t}\n}'
        try:
            return json.loads(prompt.removeprefix(original_prompt))
        except:
            if return_raw:
                return dict(generated_text = prompt.removeprefix(original_prompt))
            else:
                return dict(name='direct_response')
    
    def tool_call_input(self, llm: Type[BaseLLM], messages: List[Dict[str, str]], prompt_template: Optional[PromptTemplate] = None, **kwargs) -> Dict[str, Any]:
        """Generate a dictionary of the tool to use and the input arguments.

        Args:
            llm (Type[BaseLLM]): LLM for the function call generation.
            messages (List[Dict[str, str]]): List of messages of the conversation history, msut contain the function metadatas.
            prompt_template (Optional[PromptTemplate], optional): Prompt template to use. If none is give, the default prompt template for the llm will be used. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary containing the name of the function and the input arguments.
        """
        # set temperature to zero if temperature not given in kwargs
        if 'temperature' not in kwargs.keys():
            kwargs['temperature'] = 0
        # Start tool selection logic
        if ((len(self.enabled_tools) == 1) & (self.enabled_tools[0] == 'direct_response')):
            return dict(name='direct_response')
        prompt_template = llm.core.prompt_template if prompt_template is None else prompt_template
        # check if the function metadata content is in the messages.
        if not any(self.function_metadata['content']  in content for content in list(map(lambda x: x['content'], messages))):
            return dict(name='direct_response')
        original_prompt = prompt_template.create_custom_prompt_with_open_role(messages=messages, end_role='function_call')
        return self.structured_input_generation(raw_prompt=original_prompt, llm=llm, **kwargs)

    def tool_call_output(self, tool_input: Dict[str, Any], return_error: bool = False) -> Optional[Dict[str, Any]]:
        """Return the output of the function call along with the input ditionary.

        Args:
            tool_input (Dict[str, Any]): Inputs of the tool, including the name of the tool and the input arguments.
            return_error (bool, optional): Whether to return the error should the tool failed to execute. If False and the tool failed, None will be returned instead of the error message.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing the name of the function, the inputs, and the outputs. If None is returned, that means no tool is required and the llm should respond directly.
        """
        import json
        from copy import deepcopy
        tool_inputs = deepcopy(tool_input)
        tool_name = tool_inputs['name']
        if tool_name == 'direct_response':
            return None
        inputs = tool_inputs.get('inputs', dict())
        tool = self.tool_map[tool_name]
        try:
            output = tool(**inputs)
        except Exception as e:
            if return_error:
                tool_inputs['error'] = str(e)
                return tool_inputs
            else:
                return None
        # formatting output
        jsonable = True
        try:
            json.dumps(output)
        except:
            jsonable = False
        tool_inputs['output'] = output if jsonable else str(output)
        return tool_inputs

    def turn_on_tools(self, tools: List[str]) -> None:
        """Enable the given list of tools.

        Args:
            tools (List[str]): List of tool names.
        """
        for tool in tools:
            if tool in self.tool_names:
                self._enabled[tool] = True

    def turn_off_tools(self, tools: List[str]) -> None:
        """Disable the given list of tools.

        Args:
            tools (List[str]): List of tool names.
        """
        for tool in tools:
            if ((tool in self.tool_names) & (tool != 'direct_response')):
                self._enabled[tool] = False


                    
