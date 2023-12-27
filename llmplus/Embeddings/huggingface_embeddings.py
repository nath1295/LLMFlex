import os
from ..utils import get_config
os.environ['SENTENCE_TRANSFORMERS_HOME'] = get_config()['st_home']
from .base_embeddings import BaseEmbeddingsToolkit
from typing import Dict, Any

class HuggingfaceEmbeddingsToolkit(BaseEmbeddingsToolkit):

    def __init__(self, model_id: str, chunk_overlap_perc: float = 0.1,
                 model_kwargs: Dict[str, Any] = dict(), 
                 encode_kwargs: Dict[str, Any] = dict(normalize_embeddings=True), tokenizer_kwargs: Dict[str, Any] = dict()) -> None:
        """Initialising the Huggingface embeddings toolkit.

        Args:
            model_id (str): Model id (from Huggingface) to use.
            chunk_overlap_perc (float, optional): Number of tokens percentage overlap during text splitting. Defaults to 0.1.
            model_kwargs (Dict[str, Any], optional): Keyword arguments for the model. Defaults to dict().
            encode_kwargs (Dict[str, Any], optional): Keyword arguments for encoding. Defaults to dict(normalize_embeddings=True).
            tokenizer_kwargs (Dict[str, Any], optional): Keyword arguments for the tokenizer. Defaults to dict().
        """
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import SentenceTransformersTokenTextSplitter
        from transformers import AutoTokenizer
        self._model = HuggingFaceEmbeddings(model_name=model_id, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self._name = model_id
        self._type = 'huggingface_embeddings'
        self._embedding_size = self.embedding_model.client.get_sentence_embedding_dimension()
        self._text_splitter = SentenceTransformersTokenTextSplitter.from_huggingface_tokenizer(
            tokenizer = AutoTokenizer.from_pretrained(self.name, **tokenizer_kwargs),
            chunk_size = self.embedding_model.client.max_seq_length,
            chunk_overlap = int(self.embedding_model.client.max_seq_length * chunk_overlap_perc)
        )
