Module llmflex.Tools.tool_selection
===================================

Classes
-------

`ToolSelector(tools: List[Type[llmflex.Tools.base_tool.BaseTool]], model: llmflex.Models.Factory.llm_factory.LlmFactory, embeddings: Type[llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit], score_threshold: float = 0.7)`
:   Class to select the appropriate tool given the user input.

    ### Instance variables

    `last_score: float`
    :   The last similarity score when using get_tool.
        
        Returns:
            float: The last similarity score when using get_tool.

    `last_tool: str`
    :   The last tool picked regardless of the score.
        
        Returns:
            str: The last tool picked regardless of the score.

    `llm: Optional[llmflex.Models.Cores.base_core.BaseLLM]`
    :   LLM used for tool selection.
        
        Returns:
            Optional[BaseLLM]: LLM used for tool selection.

    `score_threshold: float`
    :   Score threshold for retrieving tools from vector database.
        
        Returns:
            float: Score threshold.

    `tools: List[llmflex.Tools.base_tool.BaseTool]`
    :   List of tools.
        
        Returns:
            List[BaseTool]: List of tools.

    `vectordb: Optional[llmflex.VectorDBs.faiss_vectordb.FaissVectorDatabase]`
    :   Vector database for that store tools information.
        
        Returns:
            Optional[FaissVectorDatabase]: Vector database for that store tools information.

    ### Methods

    `get_tool(self, user_input: str, history: Union[List[str], List[List[str]]] = [], prompt_template: Optional[llmflex.Prompts.prompt_template.PromptTemplate] = None, system: str = 'This is a conversation between a human user and a helpful AI assistant.') ‑> Optional[llmflex.Tools.base_tool.BaseTool]`
    :   Select the most appropriate tool for a given user input.
        
        Args:
            user_input (str): User input string.
            history (Union[List[str], List[List[str]]], optional): Current conversation history. Defaults to [].
            prompt_template (Optional[PromptTemplate], optional): Prompt template to format the prompt. If None is provide, the llm prompt template will be used. Defaults to None.
            system (str, optional): System message of the current conversation. Defaults to DEFAULT_SYSTEM_MESSAGE.
        
        Returns:
            Optional[BaseTool]: Selected tool or None if no tool is found or required.

    `set_score_threshold(self, score_threshold: float) ‑> None`
    :   Set a new score threshold.
        
        Args:
            score_threshold (float): New score threshold.