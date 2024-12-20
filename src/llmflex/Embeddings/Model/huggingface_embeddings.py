from .base_embeddings import BaseEmbeddings
from ...Tokenizer.huggingface_tokenizer import HuggingFaceTokenizer
import numpy as np
from typing import Optional, Dict, Any, List
try:
    import torch
    import torch.nn.functional as F
    torch_installed = True
except:
    import warnings
    torch_installed = False

class HuggingFaceEmbeddings(BaseEmbeddings):
    """A class for handling embeddings using Hugging Face models.

    This class inherits from the BaseEmbeddings class and provides methods for
    converting tokens to embeddings using Hugging Face models. It uses the
    HuggingFaceTokenizer class for tokenization and PyTorch for tensor operations.
    """
    def __init__(self,
            pretrained_model_name_or_path: str, 
            batch_size: int = 256, 
            verbose: bool = False, 
            normalize: bool = True,
            max_seq_len: Optional[int] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        ) -> None:
        """Initializes the HuggingFaceEmbeddings class.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
            batch_size (int, optional): The batch size to use for tokenization and embedding generation. Defaults to 256.
            verbose (bool, optional): Whether to print verbose output during tokenization and embedding generation. Defaults to False.
            normalize (bool, optional): Whether to normalize the embeddings. Defaults to True.
            max_seq_len (Optional[int], optional): The maximum sequence length to use for tokenization and embedding generation. Defaults to None.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the Hugging Face model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the Hugging Face tokenizer. Defaults to None.
        """
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from ...utils import get_config
        model_kwargs = dict() if model_kwargs is None else model_kwargs
        tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, cache_dir=model_kwargs.pop('cache_dir', get_config('hf_home')), **model_kwargs)
        self._hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=tokenizer_kwargs.pop('cache_dir', get_config('hf_home')), **tokenizer_kwargs)
        self._config = AutoConfig.from_pretrained(pretrained_model_name_or_path, cache_dir=model_kwargs.pop('cache_dir', get_config('hf_home')), **model_kwargs)
        tokenizer = HuggingFaceTokenizer.from_hf_tokenizer(tokenizer=self._hf_tokenizer)
        embedding_size = self._config.hidden_size
        max_seq_len = max_seq_len if max_seq_len else self._config.max_position_embeddings
        self._verbose = verbose
        self._batch_size = batch_size
        self._normalize = normalize
        self._device = model.device
        super().__init__(model=model, model_id=pretrained_model_name_or_path,  tokenizer=tokenizer, max_seq_len=max_seq_len, embedding_size=embedding_size)

    def batch_embed(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of text sequences.

        This method converts a batch of text sequences into their corresponding embeddings using the underlying model and tokenizer.
        Args:
            texts (List[str]): A sequence of text strings to embed.

        Returns:
            np.ndarray: A 2D numpy array of shape (len(text), embedding_size) containing the embeddings for each text string in the input sequence.
        """
        num_text = len(texts)
        num_batch = num_text // self._batch_size if ((num_text // self._batch_size) == (num_text / self._batch_size)) else (num_text // self._batch_size) + 1
        batches = [(i * self._batch_size, min((i + 1) * self._batch_size, num_text)) for i in range(num_batch)]
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        if self._verbose:
            from tqdm import tqdm
            batches = tqdm(batches)
        output = []
        for b_start, b_end in batches:
            tokens = self._hf_tokenizer(texts[b_start:b_end], padding=True, max_length=self.max_seq_len, truncation=True, return_tensors='pt', return_attention_mask=True).to(self._device)
            with torch.no_grad():
                embeddings = self._model(**tokens)
                embeddings = mean_pooling(model_output=embeddings, attention_mask=tokens['attention_mask'])
            if self._normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            output.append(embeddings.to('cpu').detach().numpy())
            del embeddings
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self._device.type == 'mps':
            torch.mps.empty_cache()
        output = np.concatenate(output, axis=0)
        return output

        

