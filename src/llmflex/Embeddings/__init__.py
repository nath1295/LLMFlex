from .Model.huggingface_embeddings import HuggingFaceEmbeddings
from .Model.api_embeddings import APIEmbeddings
from .hf_embedding_server import HFEmbeddingServer

__doc__ = """
This module provides the python APIs for embedding models. It includes:
- HuggingFaceEmbeddings: embedding model loaded from a HuggingFace repository.
- APIEmbeddings: embedding model from a hosted embedding model with `HFEmbeddingServer`.
- HFEmbeddingServer: a server for serving embedding models.
"""
