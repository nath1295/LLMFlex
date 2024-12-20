from __future__ import annotations
from abc import ABC, abstractmethod
from ..Schema.document import Document
from ..TextSplitter.base_splitter import BaseTextSplitter
from ..Embeddings.Model.base_embeddings import BaseEmbeddings
from pydantic import BaseModel
from datetime import datetime as dt
import numpy as np
import os
from typing import List, Optional, Dict, Union, Any, Tuple, Callable

class SearchResult(BaseModel):
    doc: Document
    doc_id: int
    score: float


class BaseVectorDatabase(ABC):
    """Base class of vector database.
    """
    def __init__(self, 
            embeddings: BaseEmbeddings, 
            text_splitter: Optional[BaseTextSplitter] = None, 
            split_text: bool = True,
            vdb_dir: Optional[str] = None,
            init_save: bool = True,
            **kwargs
        ) -> None:
        """Initializes the base vector database.

        Args:
            embeddings (BaseEmbeddings): The embedding model to use for vectorization.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
            split_text (bool, optional): Whether to split the input text into chunks using the text splitter by default. Defaults to True.
            vdb_dir (Optional[str], optional): The directory to store the vector database. Defaults to None.
            init_save (bool, optional): Whether to save the vector database while initialising it. Defaults to True.
        """
        self._embeddings = embeddings
        self._text_splitter = text_splitter
        if not self._text_splitter:
            from ..TextSplitter.sentence_splitter import SentenceTextSplitter
            self._text_splitter = SentenceTextSplitter(tokenizer=self._embeddings.tokenizer)
        self._split_text = split_text
        self._vdb_dir = vdb_dir
        if init_save:
            self.save()

    @property
    def embeddings(self) -> BaseEmbeddings:
        """Gets the embedding model used for vectorization.
        
        Returns:
            BaseEmbeddings: The embedding model.
        """
        return self._embeddings
    
    @property
    def text_splitter(self) -> BaseTextSplitter:
        """Gets the text splitter used for splitting input text into chunks.
        
        Returns:
            BaseTextSplitter: The text splitter.
        """
        return self._text_splitter
    
    @property
    def split_text(self) -> bool:
        """Gets whether the input text is split into chunks using the text splitter by default.
        
        Returns:
            bool: True if the input text is split by default, False otherwise.
        """
        return self._split_text
    
    @property
    def vdb_dir(self) -> Optional[str]:
        """Gets the directory to store the vector database.

        Returns:
            Optional[str]: The directory path, or None if not set.
        """
        if self._vdb_dir:
            os.makedirs(self._vdb_dir, exist_ok=True)
        return self._vdb_dir
    
    @property
    def info(self) -> Dict[str, Any]:
        """Returns a dictionary containing information about the vector database.

        Returns:
            Dict[str, Any]: A dictionary containing information about the vector database.
        """
        if not hasattr(self, '_info'):
            self._info = dict(
                vdb_class=self.__class__.__name__,
                embeddings=self.embeddings.model_id,
                embedding_size=self.embeddings.embedding_size
            )
        return self._info
    
    @property
    def data(self) -> np.ndarray:
        """Gets the list of documents stored in the vector database.

        Returns:
            np.ndarray: A list of documents stored in the vector database.
        """
        if not hasattr(self, '_data'):
            self._data = np.array([], dtype=Document)
        return self._data
    
    @property
    def doc_ids(self) -> np.ndarray:
        """Gets the unique identifiers of the documents stored in the vector database.

        Returns:
            np.ndarray: A 1D NumPy array containing the unique identifiers of the documents.
        """
        if not hasattr(self, '_doc_ids'):
            self._doc_ids = np.array([], dtype=np.int32)
        return self._doc_ids
    
    @property
    def size(self) -> int:
        """Gets the number of documents stored in the vector database.

        Returns:
            int: The number of documents stored in the vector database.
        """
        return int(self.doc_ids.shape[0])
    
    @property
    @abstractmethod
    def vectors(self) -> np.ndarray:
        """Gets the vectors of the documents stored in the vector database.

        Returns:
            np.ndarray: A 2D NumPy array containing the vectors of the documents, where each row represents a document vector.
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_exist(cls,
            embeddings: BaseEmbeddings,
            vdb_dir: str,
            text_splitter: Optional[BaseTextSplitter] = None, 
            split_text: bool = True,
            **kwargs
        ) -> BaseVectorDatabase:
        """Creates a new vector database instance from an existing vector database.

        This method is used to load a vector database that has already been saved to disk.

        Args:
            embeddings (BaseEmbeddings): The embedding model to use for vectorization.
            vdb_dir (str): The directory path to the existing vector database.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
            split_text (bool, optional): Whether to split the input text into chunks using the text splitter by default. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the vector database constructor.

        Returns:
            BaseVectorDatabase: A new vector database instance loaded from the existing vector database.
        """
        pass
    
    def _save_info(self) -> None:
        """Saves the information about the vector database to a file.
        """
        if self.vdb_dir:
            import json
            with open(os.path.join(self.vdb_dir, 'info.json'), 'w') as f:
                json.dump(self.info, f, indent=4)

    def _save_data(self) -> None:
        """Saves the data of the vector database to a file.
        """
        if self.vdb_dir:
            import pickle
            data_dict = [doc.model_dump() for doc in self.data]
            with open(os.path.join(self.vdb_dir, 'data.pkl'), 'wb') as f:
                pickle.dump(data_dict, f)
            with open(os.path.join(self.vdb_dir, 'doc_ids.pkl'), 'wb') as f:
                pickle.dump(self.doc_ids, f)

    @abstractmethod
    def _save_vectors(self) -> None:
        """Save the vectors to a file.
        """
        pass

    def save(self) -> None:
        """Save everything of the vector database.
        """
        self._save_data()
        self._save_info()
        self._save_vectors()

    def export_to_dir(self, export_dir: str) -> None:
        """Save the vector database in the given directory.

        Args:
            export_dir (str): Directory to save.
        """
        current_dir = self._vdb_dir
        self._vdb_dir = export_dir
        self.save()     
        self._vdb_dir = current_dir

    @abstractmethod
    def _add(self, vectors: np.ndarray, docs: List[Document], new_ids: np.ndarray) -> None:
        """Adds new vectors, documents, and their corresponding unique identifiers to the vector database.

        Args:
            vectors (np.ndarray): A 2D NumPy array containing the new document vectors to add to the vector database.
            docs (List[Document]): A list of new documents to add to the vector database.
            new_ids (np.ndarray): A 1D NumPy array containing the unique identifiers for the new documents.
        """
        pass

    @abstractmethod
    def _drop(self, doc_ids: Union[List[int], np.array]) -> None:
        """Removes the documents with the specified unique identifiers from the vector database.

        Args:
            doc_ids (Union[List[int], np.array]): A list or NumPy array containing the unique identifiers of the documents to remove.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_vectors_by_ids(self, doc_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """Retrieves the vectors of the documents with the specified unique identifiers from the vector database.

        Args:
            doc_ids (Union[List[int], np.ndarray]): A list or NumPy array containing the unique identifiers of the documents for which to retrieve the vectors.
            
        Returns:
            np.ndarray: A 2D NumPy array containing the vectors of the specified documents, where each row represents a document vector.
        """
        pass

    def _get_new_doc_ids(self, length: int) -> np.ndarray:
        """Generates new unique identifiers for documents to be added to the vector database.

        This method generates a 1D NumPy array containing `length` unique identifiers, starting from the current maximum unique identifier plus one.

        Args:
            length (int): The number of new unique identifiers to generate.

        Returns:
            np.ndarray: A 1D NumPy array containing the new unique identifiers.
        """
        start = 0 if self.doc_ids.shape[0] == 0 else self.doc_ids.max() + 1
        new_ids = np.arange(length, dtype=np.int32) + start
        return new_ids

    def add_documents(self, docs: Union[Document, List[Document]], split_text: Optional[bool] = None, text_splitter: Optional[BaseTextSplitter] = None) -> None:
        """Adds new documents to the vector database.

        If `split_text` is True or not provided, the input documents will be split into chunks using the provided `text_splitter` before being added to the vector database.
        
        Args:
            docs (Union[Document, List[Document]]): The document(s) to add to the vector database. Can be a single `Document` object or a list of `Document` objects.
            split_text (Optional[bool], optional): Whether to split the input text into chunks using the text splitter. Defaults to None.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.
        """
        docs = [docs] if isinstance(docs, Document) else docs
        if len(docs) == 0:
            return
        split_text = self.split_text if split_text is None else split_text
        if split_text:
            text_splitter = self.text_splitter if text_splitter is None else text_splitter
            docs = text_splitter.split_documents(docs)
        vectors = self.embeddings.batch_embed([doc.text for doc in docs])
        self._add(vectors=vectors, docs=docs, new_ids=self._get_new_doc_ids(vectors.shape[0]))
        self.save()

    def add_texts(self, 
            texts: Union[str, List[str]], 
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            split_text: Optional[bool] = None,
            text_splitter: Optional[BaseTextSplitter] = None) -> None:
        """Adds new texts to the vector database.

        If `split_text` is True or not provided, the input texts will be split into chunks using the provided `text_splitter` before being added to the vector database.

        Args:
            texts (Union[str, List[str]]): The text(s) to add to the vector database. Can be a single string or a list of strings.
            metadata (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): The metadata to associate with the new documents. Can be a single dictionary or a list of dictionaries. Defaults to None.
            split_text (Optional[bool], optional): Whether to split the input text into chunks using the text splitter. Defaults to None.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting text into chunks. Defaults to None.

        Raises:
            Exception: Number of texts and number of metadata mismatch
        """
        texts = [texts] if isinstance(texts, str) else texts
        num_docs = len(texts)
        if num_docs == 0:
            return
        if metadata is None:
            metadata = [dict()] * num_docs
        elif isinstance(metadata, dict):
            metadata = [metadata] * num_docs
        if len(metadata) != num_docs:
            raise Exception('Number of texts and number of metadata mismatch.')
        docs = [Document(text=t, metadata=mt) for t, mt in zip(texts, metadata)]
        self.add_documents(docs, split_text=split_text, text_splitter=text_splitter)

    def batch_search(self,
            queries: List[str],
            top_k: int = 5,
            scope_ids: Optional[List[int]] = None,
            batch_size: int = 256,
            **kwargs     
        ) -> List[List[SearchResult]]:
        """Performs a batch search in the vector database using the provided queries.

        This method finds the top `top_k` most similar documents for each input query in the batch.

        Args:
            queries (List[str]): A list of input queries for which to perform the search.
            top_k (int, optional): The number of most similar documents to return for each input query. Defaults to 5.
            scope_ids (Optional[List[int]], optional): A list of unique identifiers of the documents to consider for the search. If provided, only documents with these identifiers will be considered. Defaults to None.
            batch_size (int, optional): The number of queries to process at a time. Defaults to 256.

        Raises:
            ValueError: If the number of queries is zero.

        Returns:
            List[List[SearchResult]]: A list of lists, where each inner list contains the top `top_k` most similar documents for the corresponding input query.
        """
        if len(queries) == 0:
            raise ValueError('Cannot provide an empty list of queries.')
        if self.size == 0:
            return [[]] * len(queries)
        if isinstance(scope_ids, list):
            scope_ids = np.array(scope_ids, dtype=np.int32)

        num_q = len(queries)
        num_batch = num_q // batch_size if (num_q // batch_size) == (num_q / batch_size) else (num_q // batch_size) + 1
        batches = [((i * batch_size), min((i + 1) * batch_size, num_q)) for i in range(num_batch)]
        doc_ids = []
        scores = []
        for b0, b1 in batches:
            q_vectors = self.embeddings.batch_embed(queries[b0:b1])
            ids, scs = self._batch_search_by_vectors(vectors=q_vectors, top_k=top_k, scope_ids=scope_ids, **kwargs)
            doc_ids.append(ids)
            scores.append(scs)
            del q_vectors
        doc_ids = np.concatenate(doc_ids, axis=0)
        scores = np.concatenate(scores, axis=0)
        indices = np.searchsorted(self.doc_ids, doc_ids)
        docs = self.data[indices]
        results = np.vectorize(lambda doc, doc_id, score: SearchResult(doc=doc, doc_id=doc_id, score=score))(docs, doc_ids, scores).tolist()
        del docs, scores, doc_ids, indices
        return results
    
    def search(self,
            query: str,
            top_k: int = 5,
            scope_ids: Optional[List[int]] = None,
            **kwargs          
        ) -> List[SearchResult]:
        """Performs a search in the vector database using the provided query.

        This method finds the top `top_k` most similar documents for the input query.

        Args:
            query (str): The input query for which to perform the search.
            top_k (int, optional): The number of most similar documents to return for the input query. Defaults to 5.
            scope_ids (Optional[List[int]], optional): A list of unique identifiers of the documents to consider for the search. If provided, only documents with these identifiers will be considered. Defaults to None.

        Returns:
            List[SearchResult]: A list of the top `top_k` most similar documents for the input query.
        """
        return self.batch_search(queries=[query], top_k=top_k, scope_ids=scope_ids, **kwargs)[0]

    def search_by_doc(self, filter_fn: Optional[Callable[[Document], bool]] = None, ids_only: bool = True, **kwargs) -> Union[List[int], Dict[int, Document]]:
        """Searches for documents in the vector database based on a filter function.

        This method finds documents that satisfy the provided filter function. If `ids_only` is True, it returns a list of document IDs that match the filter. Otherwise, it returns a dictionary mapping document IDs to their corresponding documents.

        Args:
            filter_fn (Optional[Callable[[Document], bool]], optional): A function that takes a `Document` object and returns a boolean indicating whether the document matches the filter. Defaults to None.
            ids_only (bool, optional): Whether to return only the IDs of the matching documents or the documents themselves. Defaults to True.

        Returns:
            Union[List[int], Dict[int, Document]]: A list of document IDs that match the filter if `ids_only` is True, or a dictionary mapping document IDs to their corresponding documents otherwise.
        """
        if self.size == 0:
            if ids_only:
                return []
            else:
                return dict()
        def bool_filter(doc: Document) -> bool:
            output = True
            if filter_fn:
                output = filter_fn(doc)
            if kwargs:
               for k, v in kwargs.items():
                   if doc.metadata.get(k) != v:
                       output = False
            return output
        mask = np.vectorize(bool_filter)(self.data)
        doc_ids = self.doc_ids[mask]
        if ids_only:
            return doc_ids.tolist()
        docs = self.data[mask]
        return dict(zip(doc_ids, docs))
    
    def drop_ids(self, doc_ids: List[int]) -> None:
        """Removes documents from the vector database based on their IDs.

        This method removes the documents with the provided IDs from the vector database.

        Args:
            doc_ids (List[int]): A list of document IDs to remove from the vector database.
        """
        self._drop(doc_ids=doc_ids)
        self.save()

    def clear(self) -> None:
        """Removes all documents from the vector database.

        This method removes all documents from the vector database, effectively clearing it.
        """
        self.drop_ids(self.doc_ids)

    
                        
        
        




        



