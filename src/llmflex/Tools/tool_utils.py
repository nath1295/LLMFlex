import inspect
from typing import Literal, List, Dict, Any, Optional, Callable, Union, Type

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

def get_args_types(fn: Callable) -> Dict[str, Dict[str, Any]]:
    """Get the data types of the arguments of a function or tool given the callable.

    Args:
        fn (Callable): The function or the tool.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of the arguments and their data types and descriptions.
    """
    annotations = inspect.getfullargspec(fn).annotations
    args_desc = get_args_descriptions(fn.__doc__) if fn.__doc__ is not None else dict()
    args = dict()
    for arg, arg_type in annotations.items():
        if arg in ['return', 'cls', 'self']:
            continue

        if getattr(arg_type, '__name__') in PYTHON_TO_JSON_TYPES.keys():
            args[arg] = dict(type=PYTHON_TO_JSON_TYPES[arg_type.__name__])
            if arg in args_desc:
                args[arg]['description'] = args_desc[arg]
        elif getattr(arg_type, '__name__') == 'Literal':
            dtype = getattr(type(arg_type.__args__[0]), '__name__')
            dtype = PYTHON_TO_JSON_TYPES.get(dtype, None)
            args[arg] = dict()
            if dtype is not None:
                args[arg]['type'] = dtype
                if arg in args_desc:
                    args[arg]['description'] = args_desc[arg]
                args[arg]['enum'] = list(arg_type.__args__)
            else:
                args[arg]['type'] = getattr(arg_type, '__name__', str(arg_type))
                if arg in args_desc:
                    args[arg]['description'] = args_desc[arg]
                args[arg]['enum'] = list(map(lambda x: str(x), arg_type.__args__))
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
        else:
            args[arg] = dict()
            args[arg]['type'] = getattr(arg_type, '__name__', str(arg_type))
            if arg in args_desc:
                args[arg]['description'] = args_desc[arg]
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
        args = get_args_types(fn.__call__)
    else:
        name = normalise_tool_name(fn.__name__)
        description = get_description(desc_doc) if desc_doc else ''
        required = get_required_args(fn)
        args = get_args_types(fn)
    return dict(name=name, description=description, parameters=dict(type='object', properties=args, required=required))
    
### Tool selection
class ToolSelector:
    """Class for selecting tool.
    """
    def __init__(self, tools: List[Union[Callable, Type[BaseTool]]]) -> None:
        self._tools = tools
        self._metadatas = [get_tool_metadata(tool) for tool in self.tools]
        self._tool_names = [m['name'] for m in self._metadatas]
        self._enabled = list(zip(self._tool_names, [True] * len(self._tool_names)))
    
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
        enabled_tools = filter(lambda x: x[1], self._enabled)
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
            names = [m['name'] for m in self.metadatas]
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
        

