from typing import List
from .base_splitter import BaseTextSplitter
from ..Tokenizer.base_tokenizer import BaseTokenizer

class TokenCountTextSplitter(BaseTextSplitter):
    """Text splitter that count tokens and split texts.
    """

    def __init__(self, tokenizer: BaseTokenizer,
                 chunk_size: int = 400, chunk_overlap: int = 40) -> None:
        """Initialize the TextSplitter.

        Args:
            tokenizer (BaseTokenizer): An instance of the BaseTokenizer class.
            chunk_size (int, optional): The maximum number of tokens per text chunk. Defaults to 400.
            chunk_overlap (int, optional): The number of tokens that overlaps for each subsequent chunk. Defaults to 40.
        """
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into a list of strings.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of strings resulting from the split.
        """
        token_ids = self._tokenizer.tokenize(text)
        num_tokens = len(token_ids)
        batch_size = self._chunk_size - self._chunk_overlap
        batch_num = num_tokens // batch_size if (num_tokens // batch_size) == (num_tokens / batch_size) else (num_tokens // batch_size) + 1
        batches = map(lambda x: (x * batch_size, min((x + 1) * batch_size + self._chunk_overlap, num_tokens)), range(batch_num))
        batches = map(lambda x: token_ids[x[0]:x[1]], batches)
        chunks = list(map(lambda x: self._tokenizer.detokenize(x), batches))
        return chunks