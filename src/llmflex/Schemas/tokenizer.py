from typing import Callable, List

class Tokenizer:
    """Class to tokenize and detokenize strings.
    """
    def __init__(self, tokenize_fn: Callable[[str], List[int]], detokenize_fn: Callable[[List[int]], str]) -> None:
        """Initialising the tokenizer.

        Args:
            tokenize_fn (Callable[[str], List[int]]): Function to tokenize text.
            detokenize_fn (Callable[[List[int]], str]): Function to detokenize token ids.
        """
        self._tokenize_fn = tokenize_fn
        self._detokenize_fn = detokenize_fn

    def tokenize(self, text: str)-> List[str]:
        """Tokenize the given string.

        Args:
            text (str): String to tokenize.

        Returns:
            List[str]: List of token ids.
        """
        return self._tokenize_fn(text)
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenize the given list of token ids.

        Args:
            token_ids (List[int]): List of token ids.

        Returns:
            str: Detokenized string.
        """
        return self._detokenize_fn(token_ids)
    
    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in the given string.

        Args:
            text (str): String to count.

        Returns:
            int: Number of tokens in the given string.
        """
        return len(self.tokenize(text))
    
    def split_text_on_tokens(self, text: str, chunk_size: int = 400, chunk_overlap:int = 40) -> List[str]:
        """Split text base on token numbers.

        Args:
            text (str): Text to split.
            chunk_size (int, optional): Maximum number of tokens for each chunk. Defaults to 400.
            chunk_overlap (int, optional): Overlapping number of tokens for each consecutive chunk. Defaults to 40.

        Returns:
            List[str]: List of text chunks.
        """
        token_ids = self.tokenize(text)
        num_tokens = len(token_ids)
        batch_size = chunk_size - chunk_overlap
        batch_num = num_tokens // batch_size if (num_tokens // batch_size) == (num_tokens / batch_size) else (num_tokens // batch_size) + 1
        batches = map(lambda x: (x * batch_size, min((x + 1) * batch_size + chunk_overlap, num_tokens)), range(batch_num))
        batches = map(lambda x: token_ids[x[0]:x[1]], batches)
        chunks = list(map(lambda x: self.detokenize(x), batches))
        return chunks


    