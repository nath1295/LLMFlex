from .base_embeddings import BaseEmbeddings
import numpy as np
from typing import Optional, Dict, Any, List

class APIEmbeddings(BaseEmbeddings):
    """A class for handling embeddings using an API.

    This class extends the BaseEmbeddings class and provides methods for generating embeddings using an API.
    """
    def __init__(self, 
        base_url: str, 
        batch_size: int = 256,
        verbose: bool = False,
        normalize: bool = True,
        tokenizer_name_or_path: Optional[str] = None, 
        tokenizer_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initializes the APIEmbeddings instance.
        Args:
            base_url (str): The base URL of the API to use for generating embeddings.
            batch_size (int, optional): The maximum number of items to send in a single API request. Defaults to 256.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
            normalize (bool, optional): Whether to normalize the generated embeddings. Defaults to True.
            tokenizer_name_or_path (Optional[str], optional): The name or path of the tokenizer to use for tokenizing input text. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the tokenizer. Defaults to None.
        """
        from ...Tokenizer.huggingface_tokenizer import HuggingFaceTokenizer
        import json
        from requests import get
        model = base_url.rstrip('/')
        info = json.loads(get(model + '/info').text)
        model_id = info['model_id']
        tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        tokenizer = HuggingFaceTokenizer(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_id, **tokenizer_kwargs)
        max_seq_len =info['max_seq_len']
        embedding_size = info['embedding_size']
        self._verbose = verbose
        self._batch_size = batch_size
        self._normalize = normalize
        self._base_url = model
        super().__init__(model, model_id, tokenizer, max_seq_len, embedding_size)

    def batch_embed(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of text sequences.

        This method converts a batch of text sequences into their corresponding embeddings using the underlying model and tokenizer.
        Args:
            texts (List[str]): A sequence of text strings to embed.

        Returns:
            np.ndarray: A 2D numpy array of shape (len(text), embedding_size) containing the embeddings for each text string in the input sequence.
        """
        from requests import get
        import json
        if self._verbose:
            from tqdm import tqdm

        batch_size = self._batch_size
        num_text = len(texts)
        num_batch = num_text // batch_size if (num_text // batch_size) == (num_text / batch_size) else (num_text // batch_size) + 1
        batches = [(x * batch_size, min((x + 1) * batch_size, num_text)) for x in range(num_batch)]
        if self._verbose:
            batches = tqdm(batches)

        embeddings = []
        for b0, b1 in batches:
            req_dict = dict(
                input_texts=texts[b0:b1],
                normalize=self._normalize,
                batch_size=batch_size
            )
            content = get(self._model + '/embeddings', json=req_dict).text
            embeddings += json.loads(content)

        return np.array(embeddings, dtype=np.float32)