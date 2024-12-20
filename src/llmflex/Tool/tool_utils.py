from typing import Callable, Any, Dict, Union, Annotated, Type
from .base_tool import BaseTool

def parse_docstring(docstring: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """Parses a docstring and returns a dictionary containing the description and arguments.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        Dict[str, Union[str, Dict[str, str]]]: A dictionary containing the summary and arguments.
            The 'description' key contains the description of the function, and the 'args' key contains
            a dictionary with the argument names as keys and their descriptions as values.
    """
    lines = [line.strip() for line in docstring.split('\n') if line.strip() != '']
    section = 'description'
    description = []
    args = dict()
    arg = ''
    for line in lines:
        if line == 'Args:':
            section = 'args'
        elif line == 'Raises:':
            section = 'errors'
        elif line == 'Returns:':
            section = 'returns'
        elif section == 'description':
            description.append(line)
        elif section == 'args':
            if ':' in line:
                line_split = line.split(':')
                if len(line_split) > 1:
                    arg = line_split[0].split('(')[0].strip()
                    args[arg] = ':'.join(line_split[1:]).strip()
            elif arg != '':
                args[arg] += '\n' + line
    final = dict(
        description='\n'.join(description).strip(),
        args=args
    )
    return final

def create_function_schema(fn: Union[Callable, Type[BaseTool]]) -> Dict[str, Any]:
    """Generates the json schema of the input arguments for the given function or tool.

    Args:
        fn (Union[Callable, Type[BaseTool]]): The function or tool for which to generate a json schema.

    Returns:
        Dict[str, Any]: Json schema of the input arguments for the given function or tool.
    """
    import inspect
    from pydantic import Field, create_model
    from .base_tool import FunctionTool
    is_tool = isinstance(fn, BaseTool)
    func_name = fn.name if is_tool else fn.__name__
    docstr = inspect.getdoc(fn.__call__ if is_tool else fn)
    docstr = inspect.getdoc(fn._fn) if isinstance(fn, FunctionTool) else docstr
    if docstr:
        arg_description = parse_docstring(docstr)
        arg_description = arg_description['args']
    else:
        arg_description = dict()

    arg_dict = dict()
    call_fn = fn.__call__ if is_tool else fn
    call_fn = fn._fn if isinstance(fn, FunctionTool) else call_fn
    params = inspect.signature(call_fn).parameters
    for arg, val in params.items():
        annotation = val.annotation
        if annotation == inspect._empty:
            annotation = Any
        default = val.default
        if default == inspect._empty:
            arg_dict[arg] = (Annotated[annotation, Field(description=arg_description.get(arg, None))])
        else:
            arg_dict[arg] = (Annotated[annotation, Field(description=arg_description.get(arg, None), default=default)])
            
    model = create_model(func_name, **arg_dict)
    return model.model_json_schema()

def create_openai_function(fn: Union[Callable, Type[BaseTool]]) -> Dict[str, Any]:
    """Creates an OpenAI function object from the given function or tool.

    Args:
        fn (Union[Callable, Type[BaseTool]]): The function or tool to convert to an OpenAI function.

    Returns:
        Dict[str, Any]: An OpenAI function object.
    """
    import inspect
    parameters = create_function_schema(fn)
    if 'properties' in parameters:
        for k in parameters['properties'].keys():
            parameters['properties'][k].pop('title')
    func_name = parameters.pop('title')
    docstr = inspect.getdoc(fn)
    description = parse_docstring(docstr)['description'] if docstr else None
    function_schema = dict(name=func_name)
    if description:
        function_schema['description'] = description
    function_schema['parameters'] = parameters
    return dict(type='function', function=function_schema)