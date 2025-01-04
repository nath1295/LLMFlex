from .base_splitter import BaseTextSplitter
from typing import List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..Tokenizer.base_tokenizer import BaseTokenizer

class RecursiveTextSplitter(BaseTextSplitter):
    """RecursiveTextSplitter is a text splitter that recursively splits text into smaller chunks until each chunk is below a specified maximum token length.
    """
    def __init__(self, tokenizer: "BaseTokenizer", split_strings: Optional[List[str]] = None, chunk_size: int = 400) -> None:
        """Initializes the RecursiveTextSplitter with a tokenizer, split strings, and chunk size.

        Args:
            tokenizer (BaseTokenizer): The tokenizer to use for counting tokens.
            split_strings (Optional[List[str]], optional): A list of strings to split the text. If None is given, it will be ["\\n\\n", "\\n", "?", "!", ".", ","]. Defaults to None.
            chunk_size (int, optional): The maximum number of tokens allowed in each chunk. Defaults to 400.
        """
        self._split_strings = ["\n\n", "\n", "\?", "!", "\.", ","] if split_strings is None else split_strings
        if len(self._split_strings) == 0:
            raise ValueError('Cannot provide an empty list as split strings.')
        unique_strings = []
        for string in self._split_strings:
            if string in unique_strings:
                raise ValueError('Cannot contain duplicates in split strings.')
            else:
                unique_strings.append(string)
        
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size

    def _split_with_token(self, text: str, token: str) -> List[str]:
        import re
        strings = re.split(f'({token})', text)
        real_token = token.removeprefix('\\')
        strings = [string for string in strings if ((string != '') and (string != real_token))]
        if len(strings) > 0:
            strings = [string + real_token for string in strings[:-1]] + [strings[-1]]
        return strings

    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into a list of strings.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of strings resulting from the split.
        """
        chunks = [text]
        for token in self._split_strings:
            sizes = [self._tokenizer.get_num_tokens(chunk, add_special_tokens=False) for chunk in chunks]
            if all([s <= self._chunk_size for s in sizes]):
                return chunks
            else:
                chunks = [[c] if s <= self._chunk_size else self._split_with_token(c, token) for c, s in zip(chunks, sizes)]
                chunks = sum(chunks, [])
        return chunks
        