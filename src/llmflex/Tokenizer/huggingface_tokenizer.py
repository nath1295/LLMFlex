from __future__ import annotations
from .base_tokenizer import BaseTokenizer
from typing import List, Literal
try:
    from transformers import PreTrainedTokenizerBase, AutoTokenizer
    hf_installed = True
except:
    hf_installed = False
    import warnings

class HuggingFaceTokenizer(BaseTokenizer):
    """Huggingface tokenizer class.
    """
    def __init__(self, pretrained_model_name_or_path: str, **kwargs) -> None:
        """Initialise the tokenizer. For initialisation kwargs, please pass in other keyword arguments.

        Args:
            pretrained_model_name_or_path (str): Huggingface repository name or path to the model.
        """
        if not hf_installed:
            warnings.warn(message='"transformers" not installed. To use HuggingFaceTokenizer, please install "transformers" by running "pip install transformers".')
        from ..utils import get_config
        self._hf_tokenizer = kwargs.pop('tokenizer', None)
        if not self._hf_tokenizer:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=kwargs.pop('cache_dir', get_config('hf_home')), **kwargs)

        bos_token = self._hf_tokenizer.bos_token
        eos_token = self._hf_tokenizer.eos_token
        pad_token = self._hf_tokenizer.pad_token
        bos_token_id = self._hf_tokenizer.bos_token_id
        eos_token_id = self._hf_tokenizer.eos_token_id
        pad_token_id = self._hf_tokenizer.pad_token_id
        
        super().__init__(tokenizer_type='huggingface_tokenizer', 
                bos_token=bos_token, eos_token=eos_token, pad_token=pad_token,
                bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        
        if self.hf_tokenizer.pad_token != self.pad_token:
            self.hf_tokenizer.pad_token = self.pad_token
            self.hf_tokenizer.pad_token_id = self.pad_token_id

    @classmethod
    def from_hf_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> HuggingFaceTokenizer:
        """Initialise the tokenizer from a huggingface transformers tokenizer directly.

        Args:
            tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer.

        Returns:
            HuggingFaceTokenizer: Initialised huggingface tokenizer.
        """
        return cls(pretrained_model_name_or_path='', tokenizer=tokenizer)
    
    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        """Returns the underlying HuggingFace tokenizer instance.

        Returns:
            PreTrainedTokenizerBase: The underlying HuggingFace tokenizer instance.
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
        return self.hf_tokenizer(text=texts, add_special_tokens=add_special_tokens)['input_ids']

    def batch_detokenize(self, token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Detokenize a batch of token IDs.
        Args:
            token_ids (List[List[int]]): List of token IDs to detokenize.
            skip_special_tokens (bool, optional): Whether to skip special tokens (BOS, EOS, PAD) during detokenization. Defaults to True.

        Returns:
            List[str]: List of detokenized texts for each list of token IDs in the input list.
        """
        return self.hf_tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        
    