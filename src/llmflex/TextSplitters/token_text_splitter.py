from typing import List, Callable
from .base_text_splitter import BaseTextSplitter

class TokenCountTextSplitter(BaseTextSplitter):
    """Text splitter that count tokens and split texts.
    """

    def __init__(self, encode_fn: Callable[[str], List[int]],
                 decode_fn: Callable[[List[int]], str],
                 chunk_size: int = 400, chunk_overlap: int = 40) -> None:
        """Initialise the TextSplitter.

        Args:
            encode_fn (Callable[[str], List[int]]): Function of encode a string.
            decode_fn (Callable[[List[int]], str]): Function to decode a list of token ids.
            chunk_size (int, optional): Maximum number of tokens per text chunk. Defaults to 400.
            chunk_overlap (int, optional): Numbers of tokens that overlaps for each subsequent chunks. Defaults to 40.
        """
        from ..Schemas.tokenizer import Tokenizer
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._tokenizer = Tokenizer(tokenize_fn=encode_fn, detokenize_fn=decode_fn)

    def split_text(self, text: str) -> List[str]:
        """Splitting the given text.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of split texts.
        """
        chunks = self._tokenizer.split_text_on_tokens(text=text, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        return chunks