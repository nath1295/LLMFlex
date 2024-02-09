from .base_tool import BaseTool
from ..Data.vector_database import VectorDatabase
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from typing import Type, Optional, List

class ToolSelector:
    """Class to select the appropriate tool given the user input."""

    def __init__(self, tools: List[Type[BaseTool]], embeddings: Type[BaseEmbeddingsToolkit], score_threshold: float = 0.7) -> None:
        if len(tools) == 0:
            raise ValueError(f'At least one tool must be provided.')
        self._tool = tools
        self._vectordb = VectorDatabase.from_empty(embeddings=embeddings)
        self._score_threshold = score_threshold
        for tool in self.tools:
            index = [tool.name, tool.description]
            data = dict(name=tool.name, description=tool.description)
            self.vector_db.add_texts(index, metadata=data)

    @property
    def tools(self) -> List[Type[BaseTool]]:
        """List of tools.

        Returns:
            List[Type[BaseTool]]: List of tools.
        """

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
        if tool['score'] >= self.score_threshold:
            tool = list(filter(lambda x: x.name == tool['name'], self.tools))[0]
            return tool
        else:
            return None

