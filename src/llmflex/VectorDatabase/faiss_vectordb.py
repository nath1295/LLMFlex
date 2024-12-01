from __future__ import annotations
from .base_vectordb import BaseVectorDatabase
from ..Embeddings.Model.base_embeddings import BaseEmbeddings
from ..TextSplitter.base_splitter import BaseTextSplitter
from ..Schema.document import Document
import numpy as np
import gc
import os
from typing import Union, Optional, List, Tuple

class FaissVectorDatabase(BaseVectorDatabase):
    """Vector database class using Faiss to power. 
    """

    def __init__(self, embeddings: BaseEmbeddings, 
            text_splitter: Optional[BaseTextSplitter] = None, split_text: bool = True, vdb_dir: Optional[str] = None, init_save: bool = True, **kwargs) -> None:
        """Initializes the vector database.

        Args:
            embeddings (BaseEmbeddings): The embedding model to use for vectorization.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
            split_text (bool, optional): Whether to split the input text into chunks using the text splitter by default. Defaults to True.
            vdb_dir (Optional[str], optional): The directory to store the vector database. Defaults to None.
            init_save (bool, optional): Whether to save the vector database while initialising it. Defaults to True.
        """
        try:
            import faiss
        except:
            raise ModuleNotFoundError('"faiss" not installed. Please install with `pip install faiss-cpu`.')
        super().__init__(embeddings, text_splitter, split_text, vdb_dir, init_save, **kwargs)


    @property
    def vectors(self) -> np.ndarray:
        """Gets the vectors of the documents stored in the vector database.

        Returns:
            np.ndarray: A 2D NumPy array containing the vectors of the documents, where each row represents a document vector.
        """
        if not hasattr(self, '_index'):
            from faiss import IndexFlatL2
            self._index = IndexFlatL2(self.embeddings.embedding_size)
        vecs = self._index.reconstruct_n()
        return np.array(vecs, dtype=np.float32)
    
    @classmethod
    def from_exist(cls,
            embeddings: BaseEmbeddings,
            vdb_dir: str,
            text_splitter: Optional[BaseTextSplitter] = None, 
            split_text: bool = True,
            **kwargs
        ) -> FaissVectorDatabase:
        """Creates a new vector database instance from an existing vector database.

        This method is used to load a vector database that has already been saved to disk.

        Args:
            embeddings (BaseEmbeddings): The embedding model to use for vectorization.
            vdb_dir (str): The directory path to the existing vector database.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
            split_text (bool, optional): Whether to split the input text into chunks using the text splitter by default. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the vector database constructor.
            
        Returns:
            FaissVectorDatabase: A new vector database instance loaded from the existing vector database.
        """
        import json
        if not os.path.exists(vdb_dir):
            raise FileNotFoundError(f'Vector database directory does not exist.')
        with open(os.path.join(vdb_dir, 'info.json'), 'r') as f:
            info = json.load(f)
        if cls.__name__ != info['vdb_class']:
            old_class = info['vdb_class']
            raise TypeError(f'The existing vector database is a class "{old_class}" object. It is not compatible with "{cls.__name__}".')
        if embeddings.model_id != info['embeddings']:
            old_model_id = info['embeddings']
            raise ValueError(f'The existing vector database uses the embedding model "{old_model_id}". Please use the correct model to load the vector database.')
        if embeddings.embedding_size != info['embedding_size']:
            esize = info['embedding_size']
            raise ValueError(f'The existing vector database uses the embedding model "{old_model_id}" with embedding size {esize}. Please use the correct model to load the vector database.')
        try:
            from faiss import read_index
        except:
            raise ModuleNotFoundError('"faiss" not installed. Please install with `pip install faiss-cpu`.')        

        vdb = cls(embeddings=embeddings, vdb_dir=vdb_dir, text_splitter=text_splitter, split_text=split_text, init_save=False)
        
        import pickle
        with open(os.path.join(vdb.vdb_dir, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
            vdb._data = np.array([Document(**args) for args in data])
        with open(os.path.join(vdb.vdb_dir, 'doc_ids.pkl'), 'rb') as f:
            vdb._doc_ids = pickle.load(f)
        with open(os.path.join(vdb.vdb_dir, 'vectors.pkl'), 'rb') as f:
            vdb._index = read_index(os.path.join(vdb.vdb_dir, 'index.faiss'))
        return vdb

    def _save_vectors(self) -> None:
        """Save the vectors to a file.
        """
        if self.vdb_dir:
            from faiss import write_index
            if not hasattr(self, '_index'):
                self.vectors
            write_index(self._index, os.path.join(self.vdb_dir, 'index.faiss'))

    def _add(self, vectors: np.ndarray, docs: List[Document], new_ids: np.ndarray) -> None:
        """Adds new vectors, documents, and their corresponding unique identifiers to the vector database.

        Args:
            vectors (np.ndarray): A 2D NumPy array containing the new document vectors to add to the vector database.
            docs (List[Document]): A list of new documents to add to the vector database.
            new_ids (np.ndarray): A 1D NumPy array containing the unique identifiers for the new documents.
        """
        from faiss import normalize_L2
        if not hasattr(self, '_index'):
            self.vectors
        normalize_L2(vectors)
        self._index.add(vectors)
        self._data = np.concatenate([self.data, np.array(docs)], axis=0) if self.size > 0 else np.array(docs)
        self._doc_ids = np.concatenate([self.doc_ids, new_ids], axis=0) if self.size > 0 else new_ids

    def _drop(self, doc_ids: Union[List[int], np.array]) -> None:
        """Removes the documents with the specified unique identifiers from the vector database.

        Args:
            doc_ids (Union[List[int], np.array]): A list or NumPy array containing the unique identifiers of the documents to remove.
        """
        if not hasattr(self, '_index'):
            self.vectors
        mask = np.where(~np.isin(self.doc_ids, doc_ids))[0]
        faiss_ids = np.where(np.isin(self.doc_ids, doc_ids))[0]
        self._doc_ids = self._doc_ids[mask]
        self._vectors = self._index.remove_ids(faiss_ids)
        self._data = self._data[mask]

    def _batch_search_by_vectors(self, vectors: np.ndarray, top_k: int = 5, scope_ids: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a batch search in the vector database using the provided vectors.

        This method finds the top `top_k` most similar documents for each input vector in the batch.

        Args:
            vectors (np.ndarray): A 2D NumPy array containing the input vectors for which to perform the search. Each row represents a vector.
            top_k (int, optional): The number of most similar documents to return for each input vector. Defaults to 5.
            scope_ids (Optional[np.ndarray], optional): A 1D NumPy array containing the unique identifiers of the documents to consider for the search. If provided, only documents with these identifiers will be considered. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D NumPy arrays. The first array contains the unique identifiers of the most similar documents for each input vector, and the second array contains the similarity scores for each pair of documents.
        """
        if not hasattr(self, '_index'):
            self.vectors
        from faiss import normalize_L2, SearchParametersIVF, IDSelectorArray
        normalize_L2(vectors)
        if scope_ids is None:
            top_k = min(top_k, self._index.ntotal)
            scores, ids = self._index.search(vectors, k=top_k)
        else:
            faiss_ids = np.where(np.isin(self.doc_ids, scope_ids))[0]
            top_k = min(top_k, len(faiss_ids))
            id_selector = IDSelectorArray(faiss_ids)
            scores, ids = self._index.search(vectors, k=top_k, params=SearchParametersIVF(sel=id_selector))
        scores = 1 - scores / (2 ** 0.5)
        ids = self.doc_ids[ids]
        return ids, scores
    
    def get_vectors_by_ids(self, doc_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """Retrieves the vectors of the documents with the specified unique identifiers from the vector database.

        Args:
            doc_ids (Union[List[int], np.ndarray]): A list or NumPy array containing the unique identifiers of the documents for which to retrieve the vectors.
            
        Returns:
            np.ndarray: A 2D NumPy array containing the vectors of the specified documents, where each row represents a document vector.
        """
        doc_ids = np.array(doc_ids, dtype=np.int32) if not isinstance(doc_ids, np.ndarray) else doc_ids.astype(dtype=np.int32)
        sorted_indices = np.argsort(self.doc_ids)
        sorted_ref = self.doc_ids[sorted_indices]
        sorted_indices_in_doc = np.searchsorted(sorted_ref, doc_ids)
        indices = sorted_indices[sorted_indices_in_doc]
        vectors = list(map(lambda x: self._index.reconstruct(x), indices))
        return np.array(vectors, dtype=np.float32)
        