from abc import ABC, abstractmethod
from ...Tokenizer.base_tokenizer import BaseTokenizer
import numpy as np
from typing import Any, List

class BaseEmbeddings(ABC):
    """Base class for embedding models.
    """
    def __init__(self, model: Any, model_id: str, tokenizer: BaseTokenizer, max_seq_len: int, embedding_size: int) -> None:
        """Initializes the base embedding model.

        This method sets up the base embedding model with the given parameters.
        Args:
            model (Any): The underlying model for the embeddings.
            model_id (str): The ID of the embeddings model.
            tokenizer (BaseTokenizer): The tokenizer used to convert text into tokens.
            max_seq_len (int): The maximum sequence length that the embeddings can handle.
            embedding_size (int): The size of the embeddings.
        """
        self._model = model
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._embedding_size = embedding_size

    @property
    def model_id(self) -> str:
        """Gets the ID of the embeddings model.

        Returns:
            str: The ID of the embeddings model.
        """
        return self._model_id

    @property
    def max_seq_len(self) -> int:
        """Gets the maximum sequence length that the embeddings can handle.

        Returns:
            int: The maximum sequence length.
        """
        return self._max_seq_len
    
    @property
    def embedding_size(self) -> int:
        """Gets the size of the embeddings.

        Returns:
            int: The size of the embeddings.
        """
        return self._embedding_size
    
    @property
    def tokenizer(self) -> BaseTokenizer:
        """Gets the tokenizer used to convert text into tokens.

        Returns:
            BaseTokenizer: The tokenizer used to convert text into tokens.
        """
        return self._tokenizer
    
    @abstractmethod
    def batch_embed(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of text sequences.

        This method converts a batch of text sequences into their corresponding embeddings using the underlying model and tokenizer.
        Args:
            texts (List[str]): A sequence of text strings to embed.

        Returns:
            np.ndarray: A 2D numpy array of shape (len(text), embedding_size) containing the embeddings for each text string in the input sequence.
        """
        pass

    def embed(self, text: str) -> np.ndarray:
        """Embeds a single text sequence.

        This method converts a single text string into its corresponding embedding using the underlying model and tokenizer.
        Args:
            text (str): A single text string to embed.

        Returns:
            np.ndarray: A 1D numpy array of shape (embedding_size,) containing the embedding for the input text string.
        """
        return self.batch_embed([text])[0]