from typing import List, Optional
from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """Base class for tokenizers.
    """

    def __init__(self,
            tokenizer_type: str,
            eos_token: Optional[str], 
            bos_token: Optional[str],
            pad_token: Optional[str],
            eos_token_id: Optional[int],
            bos_token_id: Optional[int],
            pad_token_id: Optional[int]
        ) -> None:
        """Initialize the tokenizer.

        Args:
            tokenizer_type (str): The type of the tokenizer.
            eos_token (Optional[str]): The end-of-sequence token if the tokenizer is for an LLM model.
            bos_token (Optional[str]): The beginning-of-sequence token if the tokenizer is for an LLM model.
            pad_token (Optional[str]): The padding token for the tokenizer.
            eos_token_id (Optional[int]): The end-of-sequence token ID if the tokenizer is for an LLM model.
            bos_token_id (Optional[int]): The beginning-of-sequence token ID if the tokenizer is for an LLM model.
            pad_token_id (Optional[int]): The padding token ID for the tokenizer.
        """
        self._tokenizer_type = tokenizer_type
        self._bos_token = None if (bos_token is None) or (bos_token == '') else bos_token
        self._eos_token = None if (eos_token is None) or (eos_token == '') else eos_token
        self._pad_token = bos_token if (pad_token is None) or (pad_token == '') else pad_token
        self._bos_token_id = None if (bos_token is None) or (bos_token == '') else bos_token_id
        self._eos_token_id = None if (eos_token is None) or (eos_token == '') else eos_token_id
        self._pad_token_id = self._bos_token_id if (pad_token is None) or (pad_token == '') else pad_token_id
    
    @property
    def tokenizer_type(self) -> str:
        """Class type of tokenizer.

        Returns:
            str: Class type of tokenizer.
        """
        return self._tokenizer_type
    
    @property
    def eos_token(self) -> Optional[str]:
        """EOS token text.

        Returns:
            Optional[str]: EOS token text.
        """
        return self._eos_token
    
    @property
    def bos_token(self) -> Optional[str]:
        """BOS token text.

        Returns:
            Optional[str]: BOS token text.
        """
        return self._bos_token
    
    @property
    def pad_token(self) -> Optional[str]:
        """Pad token text.

        Returns:
            Optional[str]: Pad token text.
        """
        return self._pad_token
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """EOS token id.

        Returns:
            Optional[int]: EOS token id.
        """
        return self._eos_token_id
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """BOS token id.

        Returns:
            Optional[int]: BOS token id.
        """
        return self._bos_token_id
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Pad token id.

        Returns:
            Optional[int]: Pad token id.
        """
        return self._pad_token_id
    
    @abstractmethod
    def batch_tokenize(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Tokenize a batch of texts.
        Args:
            texts (List[str]): List of texts to tokenize.
            add_special_tokens (bool, optional): Whether to add special tokens (BOS, EOS, PAD) to the tokenized texts. Defaults to True.

        Returns:
            List[List[int]]: List of token IDs for each text in the input list.
        """
        pass

    @abstractmethod
    def batch_detokenize(self, token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Detokenize a batch of token IDs.
        Args:
            token_ids (List[List[int]]): List of token IDs to detokenize.
            skip_special_tokens (bool, optional): Whether to skip special tokens (BOS, EOS, PAD) during detokenization. Defaults to True.

        Returns:
            List[str]: List of detokenized texts for each list of token IDs in the input list.
        """
        pass

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Tokenize a string.

        Args:
            text (str): String to tokenize.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.

        Returns:
            List[int]: List of token IDs.
        """
        return self.batch_tokenize(texts=[text], add_special_tokens=add_special_tokens)[0]

    def detokenize(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token ids to the original text.

        Args:
            token_ids (List[int]): List of token IDs.
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.

        Returns:
            str: Decoded text.
        """
        return self.batch_detokenize(token_ids=[token_ids], skip_special_tokens=skip_special_tokens)[0]

    def get_num_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        """Get the number of tokens.

        Args:
            text (str): String to count tokens.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.

        Returns:
            int: Number of tokens.
        """
        token_ids = self.tokenize(text=text, add_special_tokens=add_special_tokens)
        return len(token_ids)

