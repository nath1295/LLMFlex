from __future__ import annotations
from .base_tokenizer import BaseTokenizer
from typing import List, Literal
try:
    from tiktoken.core import Encoding
    tt_installed = True
except:
    tt_installed = False

class OpenAITokenizer(BaseTokenizer):
    """OpenAI tokenizer class.
    """
    def __init__(self, model_id: str, **kwargs) -> None:
        """Initialize the tokenizer with the given model ID.

        Args:
            model_id (str): The ID of the OpenAI model to use for tokenization.
        """
        if not tt_installed:
            raise ModuleNotFoundError(f'"tiktoken" not installed. Please install with `pip install tiktoken`.')
        import tiktoken
        self._openai_tokenizer = kwargs.pop('tokenizer', None)
        if not self._openai_tokenizer:
            self._openai_tokenizer = tiktoken.encoding_for_model(model_id)

        bos_token_id = None
        eos_token_id = self.openai_tokenizer.eot_token
        pad_token_id = None
        bos_token = None
        eos_token = self.openai_tokenizer.decode([eos_token])
        pad_token = None
        
        super().__init__(tokenizer_type='openai_tokenizer', 
                bos_token=bos_token, eos_token=eos_token, pad_token=pad_token,
                bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

    @classmethod
    def from_hf_tokenizer(cls, tokenizer: Encoding) -> OpenAITokenizer:
        """Initialise the tokenizer from a tiktoken tokenizer directly.

        Args:
            tokenizer (Encoding): Tiktoken tokenizer.

        Returns:
            OpenAITokenizer: Initialised OpenAI tokenizer.
        """
        return cls(model_id='', tokenizer=tokenizer)
    
    @property
    def openai_tokenizer(self) -> Encoding:
        """Returns the underlying tiktoken tokenizer instance.

        Returns:
            PreTrainedTokenizerBase: The underlying tiktoken tokenizer instance.
        """
        return self._hf_tokenizer
    
    def batch_tokenize(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Tokenize a batch of texts.
        Args:
            texts (List[str]): List of texts to tokenize.
            add_special_tokens (bool, optional): Whether to add special tokens (BOS, EOS, PAD) to the tokenized texts. Defaults to True.

        Returns:
            List[List[int]]: List of token IDs for each text in the input list.
        """
        return self.openai_tokenizer.encode_batch(text=texts, allowed_special='all' if add_special_tokens else set())

    def batch_detokenize(self, token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Detokenize a batch of token IDs.
        Args:
            token_ids (List[List[int]]): List of token IDs to detokenize.
            skip_special_tokens (bool, optional): Whether to skip special tokens (BOS, EOS, PAD) during detokenization. Defaults to True.

        Returns:
            List[str]: List of detokenized texts for each list of token IDs in the input list.
        """
        return self.openai_tokenizer.decode_batch(token_ids)
        
    