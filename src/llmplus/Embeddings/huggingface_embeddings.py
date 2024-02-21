import os
from .base_embeddings import BaseEmbeddingsToolkit
from typing import Optional, Dict, Any

class HuggingfaceEmbeddingsToolkit(BaseEmbeddingsToolkit):

    def __init__(self, model_id: str, chunk_size: Optional[int] = None, chunk_overlap_perc: float = 0.1,
                 model_kwargs: Dict[str, Any] = dict(), 
                 encode_kwargs: Dict[str, Any] = dict(normalize_embeddings=True, batch_size=128)) -> None:
        """Initialising the Huggingface embeddings toolkit.

        Args:
            model_id (str): Model id (from Huggingface) to use.
            chunk_size (Optional[int], optional): Chunk size for the text splitter. If not provided, the min of the model max_seq_length or 512 will be used. Defaults to None.
            chunk_overlap_perc (float, optional): Number of tokens percentage overlap during text splitting. Defaults to 0.1.
            model_kwargs (Dict[str, Any], optional): Keyword arguments for the model. Defaults to dict().
            encode_kwargs (Dict[str, Any], optional): Keyword arguments for encoding. Defaults to dict(normalize_embeddings=True).
        """
        from ..utils import get_config
        from ..TextSplitters.token_text_splitter import TokenCountTextSplitter
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = get_config()['st_home']
        os.environ['HF_HOME'] = get_config()['hf_home']
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
        from langchain.text_splitter import SentenceTransformersTokenTextSplitter
        self._model = HuggingFaceEmbeddings(model_name=model_id, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self._name = model_id
        self._type = 'huggingface_embeddings'
        self._embedding_size = self.embedding_model.client.get_sentence_embedding_dimension()
        chunk_size = min(self.embedding_model.client.max_seq_length, 512) if not isinstance(chunk_size, int) else chunk_size
        encode_fn = lambda x: self.embedding_model.client.encode(x, add_special_tokens=False)
        decode_fn = lambda x: self.embedding_model.client.decode(x, skip_special_tokens=True)
        self._text_splitter = TokenCountTextSplitter(encode_fn=encode_fn, decode_fn=decode_fn, chunk_overlap=int(chunk_size * chunk_overlap_perc), chunk_size=chunk_size)
