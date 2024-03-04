import os
from .base_embeddings import BaseEmbeddingsToolkit, BaseEmbeddings
from typing import Optional, Dict, Any, List

class HuggingFaceEmbeddings(BaseEmbeddings):
    """Embeddings model from HuggingFace using sentence transformers."""
    def __init__(self, model_id: str, model_kwargs: Dict[str, Any] = dict(), encode_kwargs: Dict[str, Any] = dict(normalize_embeddings=True, batch_size=128)) -> None:
        from ..utils import get_config
        from sentence_transformers import SentenceTransformer
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = get_config()['st_home']
        os.environ['HF_HOME'] = get_config()['hf_home']
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self._model = SentenceTransformer(model_id, **model_kwargs)
        self._name = model_id
        self._tokenizer = self._model.tokenizer
        self._max_seq_length = self._model.max_seq_length
        self._embedding_size = self._model.get_sentence_embedding_dimension()
        self._encode_kwargs = encode_kwargs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embedded vectors.
        """
        embeddings = self._model.encode(texts, **self._encode_kwargs).tolist()
        return embeddings

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
        from ..TextSplitters.token_text_splitter import TokenCountTextSplitter
        embedding_model = HuggingFaceEmbeddings(model_name=model_id, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        name = model_id
        type = 'huggingface_embeddings'
        chunk_size = min(embedding_model._max_seq_length, 512) if not isinstance(chunk_size, int) else chunk_size
        encode_fn = lambda x: embedding_model._tokenizer.encode(x, add_special_tokens=False)
        decode_fn = lambda x: embedding_model._tokenizer.decode(x, skip_special_tokens=True)
        text_splitter = TokenCountTextSplitter(encode_fn=encode_fn, decode_fn=decode_fn, chunk_overlap=int(chunk_size * chunk_overlap_perc), chunk_size=chunk_size)
        super().__init__(embedding_model = embedding_model, text_splitter = text_splitter, name = name, 
                         type = type, embedding_size = embedding_model._embedding_size, max_seq_length = embedding_model._max_seq_length)
