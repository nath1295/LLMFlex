from __future__ import annotations
from .base_tokenizer import BaseTokenizer
from typing import List, Optional
import os
try:
    from llama_cpp import Llama
    lcpp_installed = True
except:
    lcpp_installed = False

class LlamaCppTokenizer(BaseTokenizer):
    """Llama CPP tokenizer class.
    """
    def __init__(self, pretrained_model_name_or_path: str, model_file: Optional[str] = None, **kwargs) -> None:
        """Initialise the tokenizer.

        Args:
            pretrained_model_name_or_path (str): Huggingface repository name or full path of the model of the model file.
            model_file (str, optional): Model filename if a HuggingFace repository name is given for tokenizer_id_or_path. Defaults to None.
        """
        if not lcpp_installed:
            raise ModuleNotFoundError('"llama-cpp-python" not installed. To use LlamaCppTokenizer, please install "llama-cpp-python" by running "pip install llama-cpp-python".')
        from ..utils import get_config, download_file_from_repo
        self._llama_tokenizer = kwargs.pop('llama_model', None)
        if self._llama_tokenizer is None:
            is_file = pretrained_model_name_or_path.endswith('.gguf')
            if model_file:
                file_dir = os.path.join(pretrained_model_name_or_path, model_file) if not is_file else pretrained_model_name_or_path
            else:
                file_dir = pretrained_model_name_or_path
            file_exist = os.path.exists(file_dir)
            if file_exist:
                self._llama_tokenizer = Llama(model_path=file_dir, vocab_only=True, verbose=False)
            else:
                model_dir = download_file_from_repo(repo_id=pretrained_model_name_or_path, filename=model_file, cache_dir=get_config('hf_home'), **kwargs)
                self._llama_tokenizer = Llama(model_path=model_dir, vocab_only=True, verbose=False)
        else:
            self._llama_tokenizer = self._llama_tokenizer

        eos_token_id = self._llama_tokenizer.token_eos()
        bos_token_id = self._llama_tokenizer.token_bos()
        eos_token = self._llama_tokenizer._model.detokenize([eos_token_id], special=True).decode() if eos_token_id is not None else None
        bos_token = self._llama_tokenizer._model.detokenize([bos_token_id], special=True).decode() if bos_token_id is not None else None
        
        # There is no pad token with llama-cpp-python (at least I cannot find it)
        super().__init__(
            tokenizer_type='llama_cpp',
            eos_token=eos_token, 
            bos_token=bos_token,
            pad_token=bos_token,
            eos_token_id=eos_token_id, 
            bos_token_id=bos_token_id,
            pad_token_id=bos_token_id)

    @classmethod
    def from_llama_model(cls, llama_model: Llama) -> LlamaCppTokenizer:
        """Initialise the tokenizer from a Llama CPP model directly.

        Args:
            llama_model (Llama): Llama CPP model.

        Returns:
            LlamaCppTokenizer: Initialised Llama CPP tokenizer.
        """
        return cls(pretrained_model_name_or_path='', llama_model=llama_model)
    
    @property
    def llama_tokenizer(self) -> Llama:
        """Get the underlying Llama tokenizer.
        
        Returns:
            Llama: The Llama tokenizer.
        """
        return self._llama_tokenizer

    def batch_tokenize(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Tokenize a batch of texts.
        Args:
            texts (List[str]): List of texts to tokenize.
            add_special_tokens (bool, optional): Whether to add special tokens (BOS, EOS, PAD) to the tokenized texts. Defaults to True.

        Returns:
            List[List[int]]: List of token IDs for each text in the input list.
        """
        return [self.llama_tokenizer._model.tokenize(text=text.encode(), add_bos=add_special_tokens, special=add_special_tokens) for text in texts]

    def batch_detokenize(self, token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Detokenize a batch of token IDs.
        Args:
            token_ids (List[List[int]]): List of token IDs to detokenize.
            skip_special_tokens (bool, optional): Whether to skip special tokens (BOS, EOS, PAD) during detokenization. Defaults to True.

        Returns:
            List[str]: List of detokenized texts for each list of token IDs in the input list.
        """
        return [self.llama_tokenizer._model.detokenize(tokens=tids, special=not skip_special_tokens).decode() for tids in token_ids]
    