from langchain.embeddings.base import Embeddings
from langchain.text_splitter import TextSplitter
from typing import Type

class BaseEmbeddingsToolkit:
    """Base class for storing the embedding model and the text splitter.
    """
    def __init__(self) -> None:
        """Initialising the embedding toolkit.
        """
        self._model = Embeddings()
        self._text_splitter = TextSplitter()
        self._name = 'base_embeddings'
        self._type = 'base_embeddings'
        self._embedding_size = 1024

    @property
    def embedding_model(self) -> Type[Embeddings]:
        """The embedding model.

        Returns:
            Embeddings: The embedding model.
        """
        return self._model
    
    @property
    def text_splitter(self) -> Type[TextSplitter]:
        """The text splitter.

        Returns:
            TextSplitter: The text splitter.
        """
        return self._text_splitter
    
    @property
    def embedding_size(self) -> int:
        """The embedding model's output dimensions.

        Returns:
            int: The embedding model's output dimensions.
        """
        return self._embedding_size
    
    @property
    def type(self) -> str:
        """Type of the embedding toolkit.

        Returns:
            str: Type of the embedding toolkit.
        """
        return self._type
    
    @property
    def name(self) -> str:
        """Name of the embedding model.

        Returns:
            str: Name of the embedding model.
        """
        return self._name
