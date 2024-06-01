Module llmflex.Tools.tool_utils
===============================

Functions
---------

    
`direct_response() ‑> str`
:   Direct response without using any tools or functions.
    
    Returns:
        str: Response.

    
`gen_string(llm: Type[llmflex.Models.Cores.base_core.BaseLLM], prompt: str, double_quote: bool = True, max_gen: int = 5, **kwargs) ‑> str`
:   Generate a valid string that can be wrapped between quotes safely. 
    
    Args:
        llm (Type[BaseLLM]): LLM for the string generation.
        prompt (str): Prompt for generating the string, should not include the open quote.
        double_quote (bool, optional): Whether to use double quote or not. The quote character will be added to the end of the original prompt. Defaults to True.
        max_gen (int, optional): In the unlikely event of the LLM keep generating without being able to generate a valid string, this set the maximum of generation the LLM can go. Defaults to 5.
    
    Returns:
        str: A valid string that can be wrapped between quotes safely. If a valid string cannot be generated, an empty string will be returned.

    
`get_args_descriptions(docstring: str) ‑> Dict[str, str]`
:   Get the description of the arguments of a function or tool given the docstring.
    
    Args:
        docstring (str): Docstring of the callable.
    
    Returns:
        Dict[str, str]: Dictionary of the arguments and their descriptions.

    
`get_args_dtypes(fn: Callable) ‑> Dict[str, Dict[str, Any]]`
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

    
`select(llm: Type[llmflex.Models.Cores.base_core.BaseLLM], prompt: str, options: List[str], stop: Optional[List[str]] = None, default: Optional[str] = None, retry: int = 3, raise_error: bool = False, **kwargs) ‑> Optional[str]`
:   Ask the LLM to make a selection of the list of options provided.
    
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

    `is_empty: bool`
    :   Whether tools except direct_response are enabled.
        
        Returns:
            bool: Whether tools except direct_response are enabled.

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

    ### Methods

    `get_tool_metadata(self, tool_name: str) ‑> Dict[str, Any]`
    :   Get the tool metadata given the tool name.
        
        Args:
            tool_name (str): Tool name.
        
        Returns:
            Dict[str, Any]: Metadata of the tool.

    `structured_input_generation(self, raw_prompt: str, llm: Type[llmflex.Models.Cores.base_core.BaseLLM], **kwargs) ‑> Dict[str, Any]`
    :   Core part of tool input generation.
        
        Args:
            raw_prompt (str): The starting prompt.
            llm (Type[BaseLLM]): LLM for generation.
        
        Returns:
            Dict[str, Any]: Dictionary containing the name of the function and the input arguments.

    `tool_call_input(self, llm: Type[llmflex.Models.Cores.base_core.BaseLLM], messages: List[Dict[str, str]], prompt_template: Optional[llmflex.Prompts.prompt_template.PromptTemplate] = None, **kwargs) ‑> Dict[str, Any]`
    :   Generate a dictionary of the tool to use and the input arguments.
        
        Args:
            llm (Type[BaseLLM]): LLM for the function call generation.
            messages (List[Dict[str, str]]): List of messages of the conversation history, msut contain the function metadatas.
            prompt_template (Optional[PromptTemplate], optional): Prompt template to use. If none is give, the default prompt template for the llm will be used. Defaults to None.
        
        Returns:
            Dict[str, Any]: Dictionary containing the name of the function and the input arguments.

    `tool_call_output(self, tool_input: Dict[str, Any], return_error: bool = False) ‑> Optional[Dict[str, Any]]`
    :   Return the output of the function call along with the input ditionary.
        
        Args:
            tool_input (Dict[str, Any]): Inputs of the tool, including the name of the tool and the input arguments.
            return_error (bool, optional): Whether to return the error should the tool failed to execute. If False and the tool failed, None will be returned instead of the error message.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing the name of the function, the inputs, and the outputs. If None is returned, that means no tool is required and the llm should respond directly.

    `turn_off_tools(self, tools: List[str]) ‑> None`
    :   Disable the given list of tools.
        
        Args:
            tools (List[str]): List of tool names.

    `turn_on_tools(self, tools: List[str]) ‑> None`
    :   Enable the given list of tools.
        
        Args:
            tools (List[str]): List of tool names.