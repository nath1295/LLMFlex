from .base_splitter import BaseTextSplitter
from ..Tokenizer.base_tokenizer import BaseTokenizer
from typing import List, Optional, Dict

class MarkdownTextSplitter(BaseTextSplitter):
    """MarkdownTextSplitter is a text splitter that recursively splits text into smaller chunks of markdown sections.
    """
    def __init__(self, tokenizer: BaseTokenizer, chunk_size: int = 400, config: Optional[Dict[str, bool]] = None) -> None:
        """Initializes the MarkdownTextSplitter with a tokenizer, split strings, and chunk size.

        Args:
            tokenizer (BaseTokenizer): The tokenizer to use for counting tokens.
            chunk_size (int, optional): The maximum number of tokens allowed in each chunk. Defaults to 400.
            config (config: Optional[Dict[str, bool]], optional): The split tokens and if they should be prefix of chunks or not. If None is given, it will be split by headers. Defaults to None.
        """
        default_config = {
            '\n# ': True,
            '\n## ': True,
            '\n### ': True,
            '\n#### ': True,
            '\n##### ': True,
            '\n###### ': True,
        }
        self._config = default_config if config is None else config
        if len(self._config) == 0:
            raise ValueError('Cannot provide an empty config.')
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size

    def _split_with_token(self, text: str, token: str, prefix: bool) -> List[str]:
        import re
        split_token = token if ' ' not in token else token.replace(' ', '\s')
        strings = re.split(f'({split_token})', text)
        real_token = token.removeprefix('\\')
        strings = [string for string in strings if ((string != '') and (string != real_token))]
        if len(strings) > 0:
            if prefix:
                strings = [strings[0]] + [real_token + string for string in strings[1:]]
            else:
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
        for token, prefix in self._config.items():
            sizes = [self._tokenizer.get_num_tokens(chunk, add_special_tokens=False) for chunk in chunks]
            if all([s <= self._chunk_size for s in sizes]):
                return chunks
            else:
                chunks = [[c] if s <= self._chunk_size else self._split_with_token(c, token, prefix) for c, s in zip(chunks, sizes)]
                chunks = sum(chunks, [])
        return chunks
        