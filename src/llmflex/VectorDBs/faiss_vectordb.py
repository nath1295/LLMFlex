from __future__ import annotations
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..TextSplitters.base_text_splitter import BaseTextSplitter
from ..Schemas.documents import Document
from .base_vectordb import BaseVectorDatabase
from typing import List, Any, Type, Optional, Tuple
import os, numpy as np

class FaissVectorDatabase(BaseVectorDatabase):

    def __init__(self, embeddings: Type[BaseEmbeddingsToolkit], name: Optional[str] = None, vectordb_dir: Optional[str] = None,
                 text_splitter: Optional[Type[BaseTextSplitter]] = None, **kwargs) -> None:
        """Initialise a vector database.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
            name (Optional[str], optional): Name of the vector database. Will be used as the directory base name of the vector database in vectordb_dir. If None is given, the vector database will not be saved. Defaults to None.
            vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Default text splitter for the vecetor database. If None is given, the embeddings toolkit text splitter will be used. Defaults to None.
        """
        super().__init__(embeddings=embeddings, name=name, vectordb_dir=vectordb_dir)

    def _get_empty_index(self) -> Any:
        """Return an empty index.

        Returns:
            Any: An empty index.
        """
        from faiss import IndexFlatL2
        index = IndexFlatL2(self.embeddings.embedding_size)
        return index

    @property
    def _index_filename(self) -> str:
        """Base name of the file for the index in the vector database directory.

        Returns:
            str: Base name of the file for the index in the vector database directory.
        """
        return 'index.faiss'

    def _save_index(self) -> None:
        """Save the index of the vector database.
        """
        from faiss import write_index
        write_index(self.index, os.path.join(self.db_dir, self._index_filename))

    def _load_index(self, index_dir: str) -> Any:
        """Load the index from an existing saved file.
        """
        from faiss import read_index
        return read_index(index_dir)

    def _add(self, vectors: np.ndarray[np.float32], docs: List[Document]) -> None:
        """Core method to add documents into the vector database.

        Args:
            vectors (np.ndarray[np.float32]): Array of vectors created by the indexes of the documents.
            docs (List[Document]): List of documents to add.
        """
        from faiss import normalize_L2
        current_size = self.index.ntotal
        add_size = vectors.shape[0]
        add_doc_dict = dict(zip(range(current_size, current_size + add_size), docs))
        normalize_L2(vectors)
        self.index.add(vectors)
        self._data.update(add_doc_dict)

    def _delete(self, ids: List[int]) -> None:
        """Core method to remove records by ids.

        Args:
            ids (List[int]): Ids to remove.
        """
        ids = np.array(ids, dtype=np.int32)
        if (ids > self.index.ntotal).sum() > 0:
            raise ValueError('Non-existence ids provided in the list of ids.')
        new_data = self.data.items()
        new_data = filter(lambda x: x[0] not in ids, new_data)
        new_data = map(lambda x: x[1], new_data)
        self.index.remove_ids(ids)
        self._data = dict(zip(range(self.index.ntotal), new_data))

    def _batch_search_with_scores(self, vectors: np.ndarray[np.float32], k: int = 5, ids_scope: Optional[List[int]] = None) -> Tuple[np.ndarray[np.float32], np.ndarray[np.int64]]:
        """Batch similarity search with multiple vectors.

        Args:
            vectors (np.ndarray[np.float32]): Array of vectors for the search.
            k (int, optional): Maximum results for each vector. Defaults to 5.
            ids_scope (Optional[List[int]], optional): The list of allowed ids to return for the similarity search. Defaults to None.

        Returns:
            Tuple[np.ndarray[np.float32], np.ndarray[np.int64]]: Tuple of scores and ids. Both matrices must be in the same shape.
        """
        from faiss import normalize_L2, SearchParametersIVF, IDSelectorArray
        normalize_L2(vectors)
        if ids_scope is None:
            scores, ids = self.index.search(vectors, k=k)
        else:
            id_selector = IDSelectorArray(ids_scope)
            k = min(k, len(ids_scope))
            scores, ids = self.index.search(vectors, k=k, params=SearchParametersIVF(sel=id_selector))
        scores = 1 - scores / (2 ** 0.5)
        return scores, ids
    
    def _get_vectors_by_ids(self, ids: List[int]) -> np.ndarray[np.float32]:
        """Get the array of vectors by ids.

        Args:
            ids (List[int]): Document ids.

        Returns:
            np.ndarray[np.float32]: Arrray of vectors.
        """
        ids_norm = np.array(ids, dtype=np.int32)
        if (ids_norm > self.index.ntotal).sum() > 0:
            raise ValueError('Non-existence ids provided in the list of ids.')
        vectors = list(map(lambda x: self.index.reconstruct(x), ids))
        return np.array(vectors, dtype=np.float32)