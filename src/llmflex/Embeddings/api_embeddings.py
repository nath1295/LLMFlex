from .base_embeddings import BaseEmbeddingsToolkit, BaseEmbeddings
from typing import Dict, Any, List, Optional
import json, requests, os

class APIEmbeddings(BaseEmbeddings):

    def __init__(self, base_url: str, encode_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initialising the embedding model instance.

        Args:
            base_url (str): URL for the api.
            encode_kwargs (Optional[Dict[str, Any]], optional): Encoding keyword arguments for the sentence transformer model. Defaults to None.
        """
        self.base_url = base_url.removeprefix('/')
        self.encode_kwargs = dict() if encode_kwargs is None else encode_kwargs
        self.info = json.loads(requests.get(self.base_url + '/info').content.decode())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embedded vectors.
        """
        show_progress = self.encode_kwargs.get('show_progress_bar', False)

        from tqdm import tqdm
        batch_size = self.encode_kwargs.get('batch_size', self.info['default_batch_size'])
        num_text = len(texts)
        num_batch = num_text // batch_size if (num_text // batch_size) == (num_text / batch_size) else (num_text // batch_size) + 1
        batches = list(map(lambda x: (x * batch_size, min((x + 1) * batch_size, num_text)), range(num_batch)))
        embeddings = []
        for b in tqdm(batches) if show_progress else batches:
            req_dict = dict(input_texts=texts[b[0]:b[1]])
            req_dict.update(self.encode_kwargs)
            content = requests.get(self.base_url + '/embeddings', json=req_dict).content.decode()
            embeddings += json.loads(content)
        return embeddings

class APIEmbeddingsToolkit(BaseEmbeddingsToolkit):

    def __init__(self, base_url: str, chunk_size: Optional[int] = None, chunk_overlap_perc: float = 0.1,
                 encode_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initialising the self-hosted Huggingface API embeddings toolkit.

        Args:
            base_url (str): Model id (from Huggingface) to use.
            chunk_size (Optional[int], optional): Chunk size for the text splitter. If not provided, the min of the model max_seq_length or 512 will be used. Defaults to None.
            chunk_overlap_perc (float, optional): Number of tokens percentage overlap during text splitting. Defaults to 0.1.
            encode_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for encoding text. If None is given, the default is normalize_embeddings=True, batch_size=128. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for loading tokenizer. Defaults to None.
        """
        from ..utils import get_config
        from ..TextSplitters.token_text_splitter import TokenCountTextSplitter
        from ..Schemas.tokenizer import Tokenizer
        os.environ['HF_HOME'] = get_config()['hf_home']
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        from transformers import AutoTokenizer
        encode_kwargs = dict(normalize_embeddings=True, batch_size=128) if encode_kwargs is None else encode_kwargs
        embedding_model = APIEmbeddings(base_url=base_url, encode_kwargs=encode_kwargs)
        name = embedding_model.info['model_id']
        type = 'api_embeddings'
        embedding_size = embedding_model.info['embedding_dimension']
        max_seq_length = embedding_model.info['max_seq_length']
        chunk_size = min(max_seq_length, 512) if not isinstance(chunk_size, int) else chunk_size
        tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
        encode_fn = lambda x: tokenizer.encode(x, add_special_tokens=False)
        decode_fn = lambda x: tokenizer.decode(x, skip_special_tokens=True)
        lf_tokenizer = Tokenizer(tokenize_fn=encode_fn, detokenize_fn=decode_fn)
        text_splitter = TokenCountTextSplitter(encode_fn=encode_fn, decode_fn=decode_fn, chunk_overlap=int(chunk_size * chunk_overlap_perc), chunk_size=chunk_size)
        super().__init__(embedding_model = embedding_model, text_splitter = text_splitter, tokenizer=lf_tokenizer, name = name, 
                         type = type, embedding_size = embedding_size, max_seq_length = max_seq_length)
