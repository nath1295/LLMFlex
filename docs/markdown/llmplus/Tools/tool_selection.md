Module llmplus.Tools.tool_selection
===================================

Classes
-------

`ToolSelector(tools: List[Type[llmplus.Tools.base_tool.BaseTool]], embeddings: Type[llmplus.Embeddings.base_embeddings.BaseEmbeddingsToolkit], score_threshold: float = 0.75)`
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

    `score_threshold: float`
    :   Score threshold for retrieving tools from vector database.
        
        Returns:
            float: Score threshold.

    `tools: List[Type[llmplus.Tools.base_tool.BaseTool]]`
    :   List of tools.
        
        Returns:
            List[Type[BaseTool]]: List of tools.

    `vectordb: llmplus.Data.vector_database.VectorDatabase`
    :   Vector database for that store tools information.
        
        Returns:
            VectorDatabase: Vector database for that store tools information.

    ### Methods

    `get_tool(self, user_input: str) ‑> Optional[Type[llmplus.Tools.base_tool.BaseTool]]`
    :   Select the most appropriate tool for a given user input.
        
        Args:
            user_input (str): User input string.
        
        Returns:
            Optional[BaseTool]: Selected tool or None if no tool is found or required.

    `set_score_threshold(self, score_threshold: float) ‑> None`
    :   Set a new score threshold.
        
        Args:
            score_threshold (float): New score threshold.