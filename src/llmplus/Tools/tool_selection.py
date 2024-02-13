from .base_tool import BaseTool
from ..Data.vector_database import VectorDatabase
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from typing import Type, Optional, List

class ToolSelector:
    """Class to select the appropriate tool given the user input."""

    def __init__(self, tools: List[Type[BaseTool]], embeddings: Type[BaseEmbeddingsToolkit], score_threshold: float = 0.75) -> None:
        if len(tools) == 0:
            raise ValueError(f'At least one tool must be provided.')
        self._tools = tools
        self._vectordb = VectorDatabase.from_empty(embeddings=embeddings)
        self._score_threshold = score_threshold
        self._last_score = 0
        self._last_tool = ''
        for tool in self.tools:
            index = [tool.name, tool.description]
            data = dict(name=tool.name, description=tool.description)
            self.vectordb.add_texts(index, metadata=data)

    @property
    def tools(self) -> List[Type[BaseTool]]:
        """List of tools.

        Returns:
            List[Type[BaseTool]]: List of tools.
        """
        return self._tools

    @property
    def vectordb(self) -> VectorDatabase:
        """Vector database for that store tools information.

        Returns:
            VectorDatabase: Vector database for that store tools information.
        """
        return self._vectordb

    @property
    def score_threshold(self) -> float:
        """Score threshold for retrieving tools from vector database.

        Returns:
            float: Score threshold.
        """
        return self._score_threshold
    
    @property
    def last_score(self) -> float:
        """The last similarity score when using get_tool.

        Returns:
            float: The last similarity score when using get_tool.
        """
        return self._last_score
    
    @property
    def last_tool(self) -> str:
        """The last tool picked regardless of the score.

        Returns:
            str: The last tool picked regardless of the score.
        """
        return self._last_tool
    
    def set_score_threshold(self, score_threshold: float) -> None:
        """Set a new score threshold.

        Args:
            score_threshold (float): New score threshold.
        """
        self._score_threshold = score_threshold

    def get_tool(self, user_input: str) -> Optional[Type[BaseTool]]:
        """Select the most appropriate tool for a given user input.

        Args:
            user_input (str): User input string.

        Returns:
            Optional[BaseTool]: Selected tool or None if no tool is found or required.
        """
        tool = self.vectordb.search(user_input, top_k=1, index_only=False)[0]
        self._last_score = tool['score']
        self._last_tool = tool['metadata']['name']
        if tool['score'] >= self.score_threshold:
            tool = list(filter(lambda x: x.name == tool['metadata']['name'], self.tools))[0]
            return tool
        else:
            return None

