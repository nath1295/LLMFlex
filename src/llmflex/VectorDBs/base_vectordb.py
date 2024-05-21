from __future__ import annotations
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..TextSplitters.base_text_splitter import BaseTextSplitter
from ..Schemas.documents import Document
from abc import abstractmethod, ABC
from typing import List, Dict, Union, Any, Type, Optional, Sequence, Callable, Tuple
import os, numpy as np

def default_vectordb_dir() -> str:
    """Default home directory of vector databases.

    Returns:
        str: Default home directory of vector databases.
    """
    from ..utils import get_config
    home = os.path.join(get_config()['package_home'], 'vector_databases')
    if not os.path.exists(home):
        os.makedirs(home)
    return home

def list_vectordbs(vectordb_dir: Optional[str] = None) -> List[str]:
    """List all the vector databases in the given directory.

    Args:
        vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.

    Returns:
        List[str]: List all the vector databases in the given directory.
    """
    vectordb_dir = default_vectordb_dir() if vectordb_dir is None else vectordb_dir
    vectordb_dir = vectordb_dir if os.path.exists(vectordb_dir) else default_vectordb_dir()
    dbs = list(filter(lambda x: os.path.isdir(os.path.join(vectordb_dir, x)), os.listdir(vectordb_dir)))
    dbs = list(filter(lambda x: os.path.exists(os.path.join(vectordb_dir, x, 'info.json')), dbs))
    return dbs

def name_checker(name: str) -> str:
    """Raise error if the given string has space, newline characters, or tab characters.

    Args:
        name (str): String to check.

    Returns:
        str: Return the given text if it passes all the checkes.
    """
    if ' ' in name:
        raise ValueError(f'Spaces cannot be in the name.')
    if '\n' in name:
        raise ValueError(f'Newline characters cannot be in the name.')
    if '\r' in name:
        raise ValueError(f'Newline characters cannot be in the name.')
    if '\t' in name:
        raise ValueError(f'Tab characters cannot be in the name.')
    return name

class BaseVectorDatabase(ABC):
    """Base class for vector databases.
    """
    def __init__(self, embeddings: Type[BaseEmbeddingsToolkit], name: Optional[str] = None, vectordb_dir: Optional[str] = None, 
                 text_splitter: Optional[Type[BaseTextSplitter]] = None, **kwargs) -> None:
        """Initialise a vector database.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
            name (Optional[str], optional): Name of the vector database. Will be used as the directory base name of the vector database in vectordb_dir. If None is given, the vector database will not be saved. Defaults to None.
            vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Default text splitter for the vecetor database. If None is given, the embeddings toolkit text splitter will be used. Defaults to None.
        """
        self._embeddings = embeddings
        self._name = name_checker(name) if name is not None else None
        vectordb_dir = default_vectordb_dir() if vectordb_dir is None else vectordb_dir
        self._db_dir = os.path.join(vectordb_dir, self.name) if self.name is not None else None
        if self.db_dir is not None:
            os.makedirs(self.db_dir, exist_ok=True)
        self._index = self._get_empty_index()
        self._data = dict()
        self._text_splitter = self.embeddings.text_splitter if text_splitter is None else text_splitter

    @property
    def embeddings(self) -> BaseEmbeddingsToolkit:
        """Embeddings toolkit used in the vector database.

        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit used in the vector database.
        """
        return self._embeddings
    
    @property
    def text_splitter(self) -> BaseTextSplitter:
        """Default text splitter for the vector database.

        Returns:
            BaseTextSplitter: Default text splitter for the vector database.
        """
        return self._text_splitter
    
    @property
    def index(self) -> Any:
        """Index of the vector database.

        Returns:
            Any: Index of the vector database.
        """
        return self._index
    
    @property
    def name(self) -> Optional[str]:
        """Name of the vector database.

        Returns:
            Optional[str]: Name of the vector database.
        """
        return self._name
    
    @property
    def info(self) -> Dict[str, Any]:
        """Information of the vector database.

        Returns:
            Dict[str, Any]: Information of the vector database.
        """
        if not hasattr(self, '_info'):
            from ..utils import current_time, read_json
            if self.db_dir is not None:
                info_dir = os.path.join(self.db_dir, 'info.json')
                if os.path.exists(info_dir):
                    self._info = read_json(info_dir)
                else:
                    self._info = dict(embeddings=self.embeddings.name, last_update=current_time())
            else:
                self._info = dict(embeddings=self.embeddings.name, last_update=current_time())
        return self._info
    
    @property
    def db_dir(self) -> Optional[str]:
        """Directory of the vector database.

        Returns:
            Optional[str]: Directory of the vector database.
        """
        return self._db_dir
    
    @property
    def data(self) -> Dict[int, Document]:
        """Dictionary of all the documents in the vector database.

        Returns:
            Dict[int, Document]: Dictionary of all the documents in the vector database.
        """
        return self._data
    
    @property
    def size(self) -> int:
        """Number of documents in the vector database.

        Returns:
            int: Number of documents in the vector database.
        """
        return len(self.data)
    
    @abstractmethod
    def _get_empty_index(self) -> Any:
        """Return an empty index.

        Returns:
            Any: An empty index.
        """
        pass

    @property
    @abstractmethod
    def _index_filename(self) -> str:
        """Base name of the file for the index in the vector database directory.

        Returns:
            str: Base name of the file for the index in the vector database directory.
        """
        pass
    
    @abstractmethod
    def _save_index(self) -> None:
        """Save the index of the vector database.
        """
        pass

    @abstractmethod
    def _load_index(self, index_dir: str) -> Any:
        """Load the index from an existing saved file.
        """
        pass

    @abstractmethod
    def _add(self, vectors: np.ndarray[np.float32], docs: List[Document]) -> None:
        """Core method to add documents into the vector database.

        Args:
            vectors (np.ndarray[np.float32]): Array of vectors created by the indexes of the documents.
            docs (List[Document]): List of documents to add.
        """
        pass

    @abstractmethod
    def _delete(self, ids: List[int]) -> None:
        """Core method to remove records by ids.

        Args:
            ids (List[int]): Ids to remove.
        """
        pass

    @abstractmethod
    def _batch_search_with_scores(self, vectors: np.ndarray[np.float32], k: int = 5, ids_scope: Optional[List[int]] = None) -> Tuple[np.ndarray[np.float32], np.ndarray[np.int64]]:
        """Batch similarity search with multiple vectors.

        Args:
            vectors (np.ndarray[np.float32]): Array of vectors for the search.
            k (int, optional): Maximum results for each vector. Defaults to 5.
            ids_scope (Optional[List[int]], optional): The list of allowed ids to return for the similarity search. Defaults to None.

        Returns:
            Tuple[np.ndarray[np.float32], np.ndarray[np.int64]]: Tuple of scores and ids. Both matrices must be in the same shape.
        """
        pass

    @abstractmethod
    def _get_vectors_by_ids(self, ids: List[int]) -> np.ndarray[np.float32]:
        """Get the array of vectors by ids.

        Args:
            ids (List[int]): Document ids.

        Returns:
            np.ndarray[np.float32]: Arrray of vectors.
        """
        pass

    @classmethod
    def from_exist(cls, embeddings: Type[BaseEmbeddingsToolkit], name: str, vectordb_dir: Optional[str] = None,
                   text_splitter: Optional[Type[BaseTextSplitter]] = None, **kwargs) -> BaseVectorDatabase:
        """Load the vector database from an existing vector database.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
            name (str): Name of the existing database.
            vectordbs_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.


        Returns:
            BaseVectorDatabase: The initialised vector database.
        """
        vectordbs_dir = default_vectordb_dir() if vectordb_dir is None else vectordb_dir
        name = name_checker(name)
        existing_dbs = list_vectordbs(vectordb_dir=vectordbs_dir)
        if name not in existing_dbs:
            raise ValueError(f'The vector database "{name}" does not exist.')
        
        from ..utils import read_json
        import pickle
        db_info_dir = os.path.join(vectordbs_dir, name, 'info.json')
        db_info = read_json(db_info_dir)
        db_embeddings_name = db_info.get('embeddings', None)
        vdb = cls(embeddings, name, vectordbs_dir, text_splitter)
        data_dir =os.path.join(vdb.db_dir, 'data.pkl')
        if os.path.exists(data_dir):
            with open(data_dir, 'rb') as f:
                vdb._data = pickle.load(f)
        else: # recovering from old format
            print('Trying to recover from old format...')
            old_data_dir = os.path.join(vdb.db_dir, 'index.pkl')
            if os.path.exists(old_data_dir):
                with open(old_data_dir, 'rb') as f:
                    data = pickle.load(f)
                data = list(map(lambda x: Document(index=x.page_content, metadata=x.metadata), data[0]._dict.values()))
                vdb.add_documents(data, split_text=False)
            else:
                raise FileExistsError(f'No raw data has been saved. Vector database cannot be recovered.')
        if db_embeddings_name == embeddings.name:
            vdb._index = vdb._load_index(os.path.join(vdb.db_dir, vdb._index_filename))
        else:
            print(f'You are using a different embeddings model. Switching from embedding model {db_embeddings_name} to {embeddings.name}.')
            vdb.add_documents(list(vdb.data.values()), split_text=False)
            vdb.info['embeddings'] = embeddings.name
        vdb.save()
        return vdb

    @classmethod
    def from_documents(cls, embeddings: Type[BaseEmbeddingsToolkit], docs: List[Document], 
                      name: Optional[str] = None, vectordb_dir: Optional[str] = None,
                      split_text: bool = True, text_splitter: Optional[Type[BaseTextSplitter]] = None, **kwargs) -> BaseVectorDatabase:
        """Load the vector database from existing documents.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
            docs (List[Document]): List of documents to use.
            name (Optional[str], optional): Name of the vector database. Will be used as the directory base name of the vector database in vectordb_dir. If None is given, the vector database will not be saved. Defaults to None.
            vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
            split_text (bool, optional): Whether to split the docuements with the embeddings toolkit text splitter. Defaults to True.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.

        Returns:
            BaseVectorDatabase: The initialised vector database.
        """
        vdb = cls(embeddings, name, vectordb_dir, text_splitter)
        vdb.add_documents(docs=docs, split_text=split_text, text_splitter=text_splitter)
        vdb.save() # In case an empty list of docs is given.
        return vdb
    
    @classmethod
    def from_texts(cls, embeddings: Type[BaseEmbeddingsToolkit], texts: List[str], metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                      name: Optional[str] = None, vectordb_dir: Optional[str] = None,
                      split_text: bool = True, text_splitter: Optional[Type[BaseTextSplitter]] = None, **kwargs) -> BaseVectorDatabase:
        """Load the vector database from existing texts.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit to use.
            texts (List[str]): List of texts to add.
            metadata (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Metadata to add along with the texts. Defaults to None.
            name (Optional[str], optional): Name of the vector database. Will be used as the directory base name of the vector database in vectordb_dir. If None is given, the vector database will not be saved. Defaults to None.
            vectordb_dir (Optional[str], optional): Directory where the vector databases live. If None is given, the default_vectordb_dir will be used. Defaults to None.
            split_text (bool, optional): Whether to split the docuements with the embeddings toolkit text splitter. Defaults to True.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.

        Returns:
            BaseVectorDatabase: The initialised vector database.
        """
        vdb = cls(embeddings, name, vectordb_dir, text_splitter)
        vdb.add_texts(texts=texts, metadata=metadata, split_text=split_text, text_splitter=text_splitter)
        vdb.save() # In case an empty list of texts is given.
        return vdb

    def _save_data(self) -> None:
        """Save the documents in the vector database.
        """
        if self.db_dir is not None:
            import pickle
            with open(os.path.join(self.db_dir, 'data.pkl'), 'wb') as f:
                pickle.dump(self.data, f)

    def _save_info(self) -> None:
        """Save information about the vector database.
        """
        from ..utils import save_json, current_time
        self.info['last_update'] = current_time()
        if self.db_dir is not None:
            save_json(self.info, os.path.join(self.db_dir, 'info.json'))

    def save(self) -> None:
        """Save the vector database.
        """
        if self.db_dir is not None:
            self._save_data()
            self._save_index()
            self._save_info()

    def add_documents(self, docs: List[Document], split_text: bool = True, text_splitter: Optional[Type[BaseTextSplitter]] = None) -> None:
        """Add documents into the vector database.

        Args:
            docs (List[Document]): List of documents to split.
            split_text (bool, optional): Whether to split the docuements with the embeddings toolkit text splitter. Defaults to True.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.
        """
        if len(docs) != 0:
            text_splitter = self.text_splitter if text_splitter is None else text_splitter
            docs = text_splitter.split_documents(docs) if split_text else docs
            vectors = list(map(lambda x: x.index, docs))
            vectors = self.embeddings.batch_embed(vectors)
            vectors = np.array(vectors, dtype=np.float32)
            self._add(vectors=vectors, docs=docs)
            self.save()

    def add_docs_with_vectors(self, vectors: Sequence[Sequence[float]], docs: List[Document]) -> None:
        """Add documents with pre-embedded vectors into the vector database.

        Args:
            vectors (Sequence[Sequence[float]]): Pre-embedded vectors.
            docs (List[Document]): List of documents.
        """
        len_vec = len(vectors)
        len_doc = len(docs)
        if len_vec != 0:
            if len_vec != len_doc:
                raise ValueError(f'{len_vec} vectors are given but the number of documents given is {len_doc}. Make sure the number of documents match with the number of vectors given.')
            vectors = np.array(vectors, dtype=np.float32)
            self._add(vectors, docs)
            self.save()

    def add_texts(self, texts: List[str], metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None, 
                  split_text: bool = True, text_splitter: Optional[Type[BaseTextSplitter]] = None) -> None:
        """Add texts into the vector database.

        Args:
            texts (List[str]): List of texts to add.
            metadata (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Metadata to add along with the texts. Defaults to None.
            split_text (bool, optional): Whether to split the docuements with the embeddings toolkit text splitter. Defaults to True.
            text_splitter (Optional[Type[BaseTextSplitter]], optional): Text splitter to split the documents. If none given, the embeddings toolkit text splitter will be used. Defaults to None.
        """
        if len(texts) != 0:
            if metadata is None:
                metadata = [dict()] * len(texts)
            if isinstance(metadata, list):
                if len(metadata) != len(texts):
                    raise ValueError('Number of texts does not match with number of metadata.')
            else:
                metadata = [metadata] * len(texts)
            docs = list(map(lambda x: Document(index=x[0], metadata=x[1]), list(zip(texts, metadata))))
            self.add_documents(docs=docs, split_text=split_text, text_splitter=text_splitter)

    def batch_search(self, queries: List[str], top_k: int = 5, index_only: bool = True,
                      batch_size: int = 100, filter_fn: Optional[Callable[[Document], bool]] = None, **kwargs) -> List[List[Union[str, Dict[str, Any]]]]:
        """Batch simlarity search on multiple queries.

        Args:
            queries (List[str]): List of queries.
            top_k (int, optional): Maximum number of results for each query. Defaults to 5.
            index_only (bool, optional): Whether to return the list of indexes only. Defaults to True.
            batch_size (int, optional): Batch size to perform similarity search. Defaults to 100.
            filter_fn (Optional[Callable[[Document], bool]], optional): The filter function to limit the scope of similarity search. Defaults to None.

        Returns:
            List[List[Union[str, Dict[str, Any]]]]: List of list of search results.
        """
        import gc
        # Filtering the scope of search
        scope = None
        if filter_fn:
            scope = filter(lambda x: filter_fn(x[1]), self.data.items())
        if kwargs:
            scope = self.data.items() if scope is None else scope
            for k, v in kwargs.items():
                scope = filter(lambda x: x[1].metadata.get(k) == v, scope)
        ids_scope = scope if scope is None else list(map(lambda x: x[0], scope))
        top_k = min(self.size, top_k) if ids_scope is None else min(self.size, top_k, len(ids_scope))
        if top_k == 0:
            return [[]] * len(queries)

        q_num = len(queries)
        batch_num = q_num // batch_size if ((q_num // batch_size) == (q_num / batch_size)) else (q_num // batch_size) + 1
        batches = list(map(lambda x: (x * batch_size, min(q_num, (x + 1) * batch_size)), range(batch_num)))
        scores = list()
        ids = list()
        for b in batches:
            qvecs = self.embeddings.batch_embed(queries[b[0]:b[1]])
            score, id = self._batch_search_with_scores(vectors=qvecs, k=top_k, ids_scope=ids_scope)
            scores.append(score)
            ids.append(id)
            del qvecs
            gc.collect()
        scores = np.concatenate(scores, axis=0)
        ids = np.concatenate(ids, axis=0)
        get_docs = np.vectorize(lambda x: self.data[x])
        get_indexes = np.vectorize(lambda x: x.index)
        get_metadatas = np.vectorize(lambda x: x.metadata)
        get_results = np.vectorize(lambda index, score, id, metadata: dict(index=index, score=score, id=id, metadata=metadata))
        docs = get_docs(ids)
        indexes = get_indexes(docs)
        metadatas = get_metadatas(docs)
        results = get_results(indexes, scores, ids, metadatas)
        if index_only:
            get_str = np.vectorize(lambda x: x['index'])
            return get_str(results).tolist()
        else:
            return results.tolist()
        
    def search(self, query: str, top_k: int = 5, index_only: bool = True, filter_fn: Optional[Callable[[Document], bool]] = None, **kwargs) -> List[Union[str, Dict[str, Any]]]:
        """Simlarity search on the given query.

        Args:
            query (str): Query for similarity search.
            top_k (int, optional): Maximum number of results. Defaults to 5.
            index_only (bool, optional): Whether to return the list of indexes only. Defaults to True.
            filter_fn (Optional[Callable[[Document], bool]], optional): The filter function to limit the scope of similarity search. Defaults to None.

        Returns:
            List[Union[str, Dict[str, Any]]]: List of search results.
        """
        return self.batch_search(queries=[query], top_k=top_k, index_only=index_only,filter_fn=filter_fn, **kwargs)[0]
            
    def search_by_metadata(self, ids_only: bool = False, filter_fn: Optional[Callable[[Document], bool]] = None, **kwargs) -> Union[List[int], Dict[int, Document]]:
        """Search documents or ids by metadata. Pass the filters on metadata as keyword arguments or pass a filter_fn.

        Args:
            ids_only (bool, optional): Whether to return a list of ids or a dictionary with the ids as keys and documents as values. Defaults to False.
            filter_fn (Optional[Callable[[Document], bool]], optional): The filter function. Defaults to None.

        Returns:
            Union[List[int], Dict[int, Document]]: List of ids or dictionary with the ids as keys and documents as values.
        """
        results = self.data.items()
        if filter_fn:
            results = filter(lambda x: filter_fn(x[1]), results)
        if kwargs:
            for k, v in kwargs.items():
                results = list(filter(lambda x: x[1].metadata.get(k) == v, results))

        if ids_only:
            return list(map(lambda x: x[0], results))
        return dict(results)
    
    def delete_by_metadata(self, filter_fn: Optional[Callable[[Document], bool]] = None, **kwargs) -> None:
        """Remove records by metadata. Pass the filters on metadata as keyword arguments or pass a filter_fn.

        Args:
            filter_fn (Optional[Callable[[Document], bool]], optional): The filter function. Defaults to None.
        """
        if ((not kwargs) and (not filter_fn)):
            raise ValueError('No keyword arguments or filter_fn are passed. Use the "clear" method to clear the entire database.')
        ids = self.search_by_metadata(ids_only=True, filter_fn=filter_fn, **kwargs)
        self._delete(ids)
        self.save()

    def clear(self) -> None:
        """Clear the entire vector database. Use it with caution.
        """
        import gc
        del self._index
        del self._data
        gc.collect()
        self._index = self._get_empty_index()
        self._data = dict()
        self.save()
