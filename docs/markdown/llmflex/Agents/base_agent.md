Module llmflex.Agents.base_agent
================================

Classes
-------

`Agent(llm: Type[llmflex.Models.Cores.base_core.BaseLLM], tools: Optional[List[Union[Callable, Type[llmflex.Tools.tool_utils.BaseTool]]]] = None, tool_selector: Optional[llmflex.Tools.tool_utils.ToolSelector] = None, prompt_template: Optional[llmflex.Prompts.prompt_template.PromptTemplate] = None, system_message: Optional[str] = None, examples: Optional[str] = None)`
:   Base agent class.
        
    
    Initialise the agent.
    
    Args:
        llm (Type[BaseLLM]): The llm of the agent.
        tools (Optional[List[Union[Callable, Type[BaseTool]]]], optional): List of tools for the agent, can be None if the tool selector is provided. Defaults to None.
        tool_selector (Optional[ToolSelector], optional): Tool selector for the agent, can be None if a list of tools is given. Defaults to None.
        prompt_template (Optional[PromptTemplate], optional): Prompt template for the agent, If None, the default prompt template of the llm will be used. Defaults to None.
        system_message (Optional[str], optional): System message for the agent. If None, the GENERIC_AGENT_SYSTEM_MESSAGE will be used. Defaults to None.
        examples (Optional[str], optional): Examples for the agent. If None, the GENERIC_EXAMPLES will be used. Defaults to None.

    ### Instance variables

    `llm: llmflex.Models.Cores.base_core.BaseLLM`
    :   The llm of the agent.
        
        Returns:
            BaseLLM: The llm of the agent.

    `log: List[llmflex.Agents.base_agent.AgentProcess]`
    :   Logs of the latest run of the agent.
        
        Returns:
            List[AgentProcess]: Logs of the latest run of the agent.

    `prompt_template: llmflex.Prompts.prompt_template.PromptTemplate`
    :   Prompt template used by the agent.
        
        Returns:
            PromptTemplate: Pormpt template used by the agent.

    `system: str`
    :   System message of the agent.
        
        Returns:
            str: System message of the agent.

    `tool_selector: llmflex.Tools.tool_utils.ToolSelector`
    :   Tool selector of the agent.
        
        Returns:
            ToolSelector: Tool selector of the agent.

    ### Methods

    `final_response(self, prompt: str, response_stream: bool = False) ‑> llmflex.Agents.base_agent.AgentProcess`
    :   Handling final direct response from the agent.
        
        Args:
            prompt (str): The full prompt for direct response.
            response_stream (bool, optional): Whether to stream the response. Defaults to False.
        
        Returns:
            AgentProcess: Agent process with the final response (Or response stream).

    `prerun_config(self, **kwargs) ‑> None`
    :   Any processes that should be run before the main action reaction loop.

    `run(self, task: str, max_iter: int = 10, verbose: bool = True, response_stream: bool = False, force_response: bool = True, **kwargs) ‑> Union[str, Iterator[str]]`
    :   Run the agent.
        
        Args:
            task (str): Task for the agent.
            max_iter (int, optional): Maximum number of thought-action iteractions. Defaults to 10.
            verbose (bool, optional): Whether to print the processes. Defaults to True.
            response_stream (bool, optional): Whether to return the final response as an iterator of text tokens. Defaults to False.
            force_response (bool, optional): When the max_iter is hit, whether to force a response. Defaults to True.
        
        Returns:
            Union[str, Iterator[str]]: Final response.

    `run_iter(self, task: str, max_iter: int = 10, verbose: bool = True, response_stream: bool = False, force_response: bool = True, **kwargs) ‑> Iterator[llmflex.Agents.base_agent.AgentProcess]`
    :   Run the agent and yield all itermediate processes.
        
        Args:
            task (str): Task for the agent.
            max_iter (int, optional): Maximum number of thought-action iteractions. Defaults to 10.
            verbose (bool, optional): Whether to print the processes. Defaults to True.
            response_stream (bool, optional): Whether to return the final response as an iterator of text tokens. Defaults to False.
            force_response (bool, optional): When the max_iter is hit, whether to force a response. Defaults to True.
        
        Yields:
            Iterator[AgentProcess]: Iterator a agent processes.

    `show(self, content: Any, role: Optional[str] = None, max_char: Optional[int] = None, end: str = '\n') ‑> None`
    :   Print the given content.
        
                Args:
                    content (Any): Content to print.
                    role (Optional[str], optional): Prefix for the content. Defaults to None.
                    max_char (Optional[int], optional): Maximum character of the content to print. Defaults to None.
                    end (str, optional): End character for the print function. Defaults to '
        '.

`AgentProcess(role: Literal['thought', 'action_input', 'action_output', 'response'], content: Any)`
:   Class for hosting agent step outputs.

    ### Instance variables

    `content: Any`
    :

    `role: Literal['thought', 'action_input', 'action_output', 'response']`
    :