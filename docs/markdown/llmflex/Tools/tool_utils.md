Module llmflex.Tools.tool_utils
===============================

Functions
---------

    
`get_args_descriptions(docstring: str) ‑> Dict[str, str]`
:   Get the description of the arguments of a function or tool given the docstring.
    
    Args:
        docstring (str): Docstring of the callable.
    
    Returns:
        Dict[str, str]: Dictionary of the arguments and their descriptions.

    
`get_args_types(fn: Callable) ‑> Dict[str, Dict[str, Any]]`
:   Get the data types of the arguments of a function or tool given the callable.
    
    Args:
        fn (Callable): The function or the tool.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of the arguments and their data types and descriptions.

    
`get_description(docstring: str) ‑> str`
:   Get the function or tool description from the docstring.
    
    Args:
        docstring (str): The docstring of the function or tool.
    
    Returns:
        str: The description of the function or tool.

    
`get_required_args(fn: Callable) ‑> List[str]`
:   Get the mandatory arguments of the given function.
    
    Args:
        fn (Callable): Function of interest.
    
    Returns:
        List[str]: List of mandatory arguments.

    
`get_tool_metadata(fn: Union[Callable, Type[llmflex.Tools.tool_utils.BaseTool]]) ‑> Dict[str, Any]`
:   Get the tool metadata as a dictionary.
    
    Args:
        fn (Union[Callable, Type[BaseTool]]): Tool or python function.
    
    Returns:
        Dict[str, Any]: Metadata of the tool or function.

    
`normalise_tool_name(name: str) ‑> str`
:   Make the tool name in lower case and split it with "_".
    
    Args:
        name (str): Original name of the tool.
    
    Returns:
        str: Normalised tool name.

Classes
-------

`BaseTool(name: Optional[str] = None, description: Optional[str] = None)`
:   Base class for tools.
        
    
    Initialising the tool.
    
    Args:
        name (Optional[str], optional): Name of the tool. If not given, it will be the tool class name. Defaults to None.
        description (Optional[str], optional): Description of the tool. If not given, it will read from the docstring of the tool class. Defaults to None.

    ### Descendants

    * llmflex.Tools.browser_tool.BrowserTool

    ### Instance variables

    `description: str`
    :   Description of the tool.
        
        Returns:
            str: Description of the tool.

    `name: str`
    :   Name of the tool.
        
        Returns:
            str: Name of the tool.

`ToolSelector(tools: List[Union[Callable, Type[llmflex.Tools.tool_utils.BaseTool]]])`
:   Class for selecting tool.

    ### Instance variables

    `enabled_tools: List[str]`
    :   List of tool names that are enabled.
        
        Returns:
            List[str]: List of tool names that are enabled.

    `function_metadata: Dict[str, str]`
    :   A message with the role "function_metadata" and content being the metadata of the tools.
        
        Returns:
            Dict[str, str]: A message with the role "function_metadata" and content being the metadata of the tools.

    `metadatas: List[Dict[str, Any]]`
    :   List of metadatas of tools.
        
        Returns:
            List[Dict[str, Any]]: List of metadatas of tools.

    `num_tools: int`
    :   Number of enabled tools.
        
        Returns:
            int: Number of enabled tools.

    `tool_map: Dict[str, Union[Callable, Type[llmflex.Tools.tool_utils.BaseTool]]]`
    :   Map of tool names and the tools.
        
        Returns:
            Dict[str, Union[Callable, Type[BaseTool]]]: Map of tool names and the tools.

    `tool_names: List[str]`
    :   List of tool names.
        
        Returns:
            List[str]: List of tool names.

    `tools: List[Union[Callable, Type[llmflex.Tools.tool_utils.BaseTool]]]`
    :   List of available tools.
        
        Returns:
            List[Union[Callable, Type[BaseTool]]]: List of available tools.