Module llmplus.Tools.base_tool
==============================

Classes
-------

`BaseTool(name: str = 'base_tool', description: str = 'This is a tool from the base tool class. It does not do anything.')`
:   This is a base class for callables for LLMs.
        
    
    Initialising the tool.

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

    ### Methods

    `run(self, tool_input: str, llm: llmplus.Models.Cores.base_core.BaseLLM, stream: bool = False, history: Optional[List[List[str]]] = None, prompt_template: Optional[llmplus.Prompts.prompt_template.PromptTemplate] = None, **kwargs) ‑> Union[str, Iterator[str]]`
    :   Run the tool and return the output as a string.
        
        Args:
            tool_input (str): String input for to run the tool.
            llm (BaseLLM): LLM to generate the output.
            stream (bool, optional): Whether to stream the output, if True, a generator of the output will be returned. Defaults to False.
            history (Optional[List[List[str]]], optional): Snippet of chat history to help running the tool if required. Defaults to None.
            prompt_template (Optional[PromptTemplate], optional): prompt_template to format the chat history. Defaults to None.
        
        Returns:
            Union[str, Iterator[str]]: Output of the tool.