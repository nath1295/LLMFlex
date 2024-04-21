from ..TextSplitters.base_text_splitter import BaseTextSplitter
from ..Schemas.tokenizer import Tokenizer
from langchain.embeddings.base import Embeddings
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Type

class BaseEmbeddings(ABC):
    """Base class for embeddings model.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embedded vectors.
        """
        pass

    def embed_query(self, text: str) -> List[float]:
        """Embed one string.

        Args:
            text (str): String to embed.

        Returns:
            List[float]: embeddings of the string. 
        """
        return self.embed_documents([text])[0]

class LangchainEmbeddings(Embeddings):
    """Class for langchain compatible embeddings.
    """
    def __init__(self, model: Type[BaseEmbeddings]) -> None:
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts=texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text=text)

class BaseEmbeddingsToolkit:
    """Base class for storing the embedding model and the text splitter.
    """
    def __init__(self, embedding_model: Type[BaseEmbeddings], text_splitter: Type[BaseTextSplitter], tokenizer: Tokenizer,
                 name: str, type: str, embedding_size: int, max_seq_length: int) -> None:
        from ..utils import validate_type
        self._embedding_model = validate_type(embedding_model, BaseEmbeddings)
        self._text_splitter = validate_type(text_splitter, BaseTextSplitter)
        self._tokenizer = validate_type(tokenizer, Tokenizer)
        self._name = validate_type(name, str)
        self._type = validate_type(type, str)
        self._embedding_size = validate_type(embedding_size, int)
        self._max_seq_length = validate_type(max_seq_length, int)

    @property
    def embedding_model(self) -> BaseEmbeddings:
        """The embedding model.

        Returns:
            BaseEmbeddings: The embedding model.
        """
        return self._embedding_model
    
    @property
    def text_splitter(self) -> BaseTextSplitter:
        """The text splitter.

        Returns:
            BaseTextSplitter: The text splitter.
        """
        return self._text_splitter
    
    @property
    def tokenizer(self) -> Tokenizer:
        """Tokenizer of the embedding model.

        Returns:
            Tokenizer: Tokenizer of the embedding model.
        """
        return self._tokenizer
    
    @property
    def embedding_size(self) -> int:
        """The embedding model's output dimensions.

        Returns:
            int: The embedding model's output dimensions.
        """
        return self._embedding_size
    
    @property
    def max_seq_length(self) -> int:
        """Maximum number of tokens used in each embedding vector.

        Returns:
            int: Maximum number of tokens used in each embedding vector.
        """
        self._max_seq_length
    
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
    
    @property
    def langchain_embeddings(self) -> LangchainEmbeddings:
        """Langchain compatible embeddings model.

        Returns:
            LangchainEmbeddings: Langchain compatible embeddings model.
        """
        return LangchainEmbeddings(self.embedding_model)
    
    def batch_embed(self, texts: List[str]) -> np.ndarray[np.float32]:
        """Embed list of texts.

        Args:
            texts (List[str]): List of text to embed.

        Returns:
            np.ndarray[np.float32]: Array of embedding vectors of the list of texts.
        """
        vectors = self.embedding_model.embed_documents(texts=texts)
        return np.array(vectors, dtype=np.float32)
    
    def embed(self, text: str) -> np.ndarray[np.float32]:
        """Embed a single string.

        Args:
            text (str): String to embed.

        Returns:
            np.ndarray[np.float32]: Vector of the embedded stirng.
        """
        return self.batch_embed([text])[0]