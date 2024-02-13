Module llmplus.Tools.base_tool
==============================

Classes
-------

`BaseTool(name: str = 'base_tool', description: str = 'This is a tool from the base tool class. It does not do anything.', verbose: bool = True)`
:   This is a base class for tools for LLMs.
        
    
    Initialising the tool.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmplus.Tools.web_search_tool.WebSearchTool

    ### Instance variables

    `description: str`
    :   Description of the tool.
        
        Returns:
            str: Description of the tool.

    `name: str`
    :   Name of the tool.
        
        Returns:
            str: Name of the tool.

    `pretty_name: str`
    :   Pretty name of the tool.
        
        Returns:
            str: Pretty name of the tool.

    ### Methods

    `print(self, text: str, **kwargs) ‑> None`
    :   Print the given text if verbose is True.
        
        Args:
            text (str): Text to print.

    `run(self, tool_input: str, llm: Type[llmplus.Models.Cores.base_core.BaseLLM], prompt_template: Optional[llmplus.Prompts.prompt_template.PromptTemplate] = None, stream: bool = False, history: Union[List[str], List[Tuple[str, str]], ForwardRef(None)] = None, add_footnote: bool = False, **kwargs) ‑> Union[str, Iterator[str]]`
    :   Run the tool and return the output as a string or an iterator of strings.
        
        Args:
            tool_input (str): String input for to run the tool.
            llm (Type[BaseLLM]): LLM to generate the output in a conversational setup.
            prompt_template (Optional[PromptTemplate], optional): prompt_template to format the chat history and create final output. If not given, the llm default prompt template will be used. Defaults to None.
            stream (bool, optional): Whether to stream the output, if True, a generator of the output will be returned. Defaults to False.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): Snippet of chat history to help running the tool if required. Defaults to None.
            add_footnote (bool, optional): Whether to append to footnote to the output. Defaults to False.
        
        Returns:
            Union[str, Iterator[str]]: Output of the tool.

    `run_with_chat(self, tool_input: str, llm: Type[llmplus.Models.Cores.base_core.BaseLLM], prompt_template: Optional[llmplus.Prompts.prompt_template.PromptTemplate] = None, stream: bool = False, history: Union[List[str], List[Tuple[str, str]], ForwardRef(None)] = None, add_footnote: bool = True, **kwargs) ‑> Iterator[Union[str, Tuple[str, str], Iterator[str]]]`
    :   Running tool with chat, it will yield the markdown friendly string of tool info for each steps and the final output, along with any extra information after the final output.
        
        Args:
            tool_input (str): String input for to run the tool.
            llm (Type[BaseLLM]): LLM to generate the output in a conversational setup.
            prompt_template (Optional[PromptTemplate], optional): prompt_template to format the chat history and create final output. If not given, the llm default prompt template will be used. Defaults to None.
            stream (bool, optional): Whether to stream the output, if True, a generator of the output will be returned. Defaults to False.
            history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): Snippet of chat history to help running the tool if required. Defaults to None.
            add_footnote (bool, optional): Whether to append to footnote to the output. Defaults to True.
        
        Yields:
            Iterator[Union[str, Iterator[str]]]: Iterator of the markdown friendly string of tool info for each steps and the final output, along with any extra information after the final output.