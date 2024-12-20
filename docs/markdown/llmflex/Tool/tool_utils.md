Module llmflex.Tool.tool_utils
==============================

Functions
---------

`create_function_schema(fn: Callable | Type[llmflex.Tool.base_tool.BaseTool]) ‑> Dict[str, Any]`
:   Generates the json schema of the input arguments for the given function or tool.
    
    Args:
        fn (Union[Callable, Type[BaseTool]]): The function or tool for which to generate a json schema.
    
    Returns:
        Dict[str, Any]: Json schema of the input arguments for the given function or tool.

`create_openai_function(fn: Callable | Type[llmflex.Tool.base_tool.BaseTool]) ‑> Dict[str, Any]`
:   Creates an OpenAI function object from the given function or tool.
    
    Args:
        fn (Union[Callable, Type[BaseTool]]): The function or tool to convert to an OpenAI function.
    
    Returns:
        Dict[str, Any]: An OpenAI function object.

`parse_docstring(docstring: str) ‑> Dict[str, str | Dict[str, str]]`
:   Parses a docstring and returns a dictionary containing the description and arguments.
    
    Args:
        docstring (str): The docstring to parse.
    
    Returns:
        Dict[str, Union[str, Dict[str, str]]]: A dictionary containing the summary and arguments.
            The 'description' key contains the description of the function, and the 'args' key contains
            a dictionary with the argument names as keys and their descriptions as values.