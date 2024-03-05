from __future__ import annotations
import os
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from langchain.schema.document import Document
from langchain.text_splitter import TextSplitter
import pandas as pd
from typing import List, Optional, Type, Dict, Any, Union

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
    vectordb_dir = vectordb_dir if ((isinstance(vectordb_dir, str)) & (os.path.exists(vectordb_dir))) else default_vectordb_dir()
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
        raise ValueError(f'Spaces cannot be in the name')
    if '\n' in name:
        raise ValueError(f'Newline characters cannot be in the name.')
    if '\r' in name:
        raise ValueError(f'Newline characters cannot be in the name.')
    if '\t' in name:
        raise ValueError(f'Tab characters cannot be in the name.')
    return name
        
def texts_to_documents(texts: List[str], 
        embeddings: Optional[Type[BaseEmbeddingsToolkit]] = None,
        text_splitter: Optional[Type[TextSplitter]] = None,
        data: Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]] = None,
        split_text: bool = True) -> List[Document]:
    """Create splitted documents from the list of text strings.

    Args:
        texts (List[str]): List of strings to split into documents.
        embeddings (Optional[Type[BaseEmbeddingsToolkit]], optional): Embedding toolkit used to split the documents. Defaults to None
        text_splitter (Optional[Type[TextSplitter]], optional): Text splitter used to split the documents. If provided, it will be used instead of the embedding toolkit. Defaults to None.
        data (Optional[Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]], optional): Metadata for each text strings. Defaults to None.
        split_text (bool, optional): Whether to split text if the given text is too long. Defaults to True.

    Returns:
        List[Document]: List of splitted documents.
    """
    if type(data) == pd.DataFrame:
        data = data.to_dict('records')
    elif type(data) == dict:
        data = [data] * len(texts)
    elif data is None:
        data = [dict()] * len(texts)
    
    docs = list(map(lambda x: Document(page_content=x[0], metadata=x[1]), list(zip(texts, data))))
    if split_text:
        if text_splitter is not None:
            docs = text_splitter.split_documents(docs)
        elif embeddings is not None:
            docs = embeddings.text_splitter.split_documents(docs)
        else:
            raise RuntimeError('Either embeddings or text_splitter has to be provided if split_text=True.')
    return docs

class VectorDatabase:
    """Vector database class, suitable for storing text data as embeddings for similarity searches and other classes that requires numerical respresentations of texts.
    """

    def __init__(self, embeddings: Type[BaseEmbeddingsToolkit], 
                 vectordb_dir: Optional[str] = None, name: Optional[str] = None,
                 save_raw: bool = False) -> None:
        """Initialising basic information of the vector database.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkits used in the vector database.
            vectordb_dir (Optional[str], optional): Parent directory of the vector database if it is not In-memory only. If None is given, the default_vectordb_dir will be used. Defaults to None.
            name (Optional[str], optional): Name of the vector database. If given, the vector database will be stored in storage. Defaults to None.
            save_raw (bool, optional): Whether to save raw text data and metadata as a separate json file. Defaults to False.
        """
        print("VectorDatabase class will be deprecated. Please use llmflex.VectorDBs.FaissVectorDatabase instead in the future.")
        self._name = '_InMemoryVectorDB_' if name is None else name_checker(name)
        self._embeddings = embeddings
        self._vectordb_dir = default_vectordb_dir() if vectordb_dir is None else os.path.abspath(vectordb_dir)
        self._save_raw = save_raw

    @property
    def name(self) -> str:
        """Name of the vector database.

        Returns:
            str: Name of the vector database.
        """
        return self._name
    
    @property
    def embeddings(self) -> Type[BaseEmbeddingsToolkit]:
        """Embeddings toolkit used in the vector database.

        Returns:
            Type[BaseEmbeddingsToolkit]: Embeddings toolkit used in the vector database.
        """
        return self._embeddings
    
    @property
    def vdb_dir(self) -> Union[str, None]:
        """Directory of the vector database if it is not in-memory only.

        Returns:
            Union[str, None]: Directory of the vector database if it is not in-memory only.
        """
        if self.name != '_InMemoryVectorDB_':
            vdb_dir = os.path.join(self._vectordb_dir, self.name)
            os.makedirs(vdb_dir, exist_ok=True)
            return vdb_dir
        else:
            return None

    @property
    def save_raw(self) -> bool:
        """Whether to save the raw data as json or not.

        Returns:
            bool: Whether to save the raw data as json or not.
        """
        return self._save_raw
    
    @property
    def info(self) -> Dict[str, Any]:
        """Information of the vector database.

        Returns:
            Dict[str, Any]: Information of the vector database.
        """
        if hasattr(self, '_info'):
            return self._info
        elif ((self.vdb_dir is not None) & ('info.json' in os.listdir(self.vdb_dir))):
            from ..utils import read_json
            self._info = read_json(os.path.join(self.vdb_dir, 'info.json'))
            return self._info
        else:
            from ..utils import save_json, current_time
            self._info = dict(embeddings=self.embeddings.name, last_update=current_time())
            if self.save_raw:
                save_json(self._info, os.path.join(self.vdb_dir, 'info.json'))
            return self._info
        
    @property
    def vectorstore(self) -> "FAISS":
        """Return the faiss vectorstore

        Returns:
            FAISS: The vector store.
        """
        return self._vectorstore
    
    @property
    def data(self) -> List[Dict[str, Any]]:
        """Raw data of the vector database.

        Returns:
            List[Dict[str, Any]]: Raw data of the vector database.
        """
        data = list(self._vectorstore.docstore._dict.values())
        data = list(map(lambda x: dict(index=x.page_content, metadata=x.metadata), data))
        return data
    
    @property
    def size(self) -> int:
        """Number of embeddings in the vector database. May be more than the number of texts you have added into the database due to text splitting for longer texts.

        Returns:
            int: Number of embeddings.
        """
        return len(self.data)
        
    def _init_vectordb(self, embeddings: Type[BaseEmbeddingsToolkit], from_exist: bool = True) -> None:
        """Initialise the langchain vectorstore

        Args:
            from_exist (bool, optional): Whether to initialise from an existing vectorstore. Defaults to True.
        """
        from langchain.vectorstores.faiss import FAISS
        import faiss
        import warnings
        warnings.filterwarnings('ignore')
        if ((from_exist) & (self.name in list_vectordbs(self._vectordb_dir))):
            if (self.info.get('embeddings', 'NOT_AVAIALBLE') == embeddings.name):
                self._vectorstore = FAISS.load_local(folder_path=self.vdb_dir, embeddings=embeddings.embedding_model)
            elif 'data.json' in os.listdir(self.vdb_dir):
                from ..utils import read_json
                data = read_json(os.path.join(self.vdb_dir, 'data.json'))
                docs = texts_to_documents(texts=list(map(lambda x: x['index'], data)), 
                                          embeddings=embeddings, data=list(map(lambda x: x['metadata'], data)), split_text=False)
                self._vectorstore = FAISS.from_documents(docs, embedding=embeddings.embedding_model)
                self.info
                self._info['embeddings'] = embeddings.name
                self.save()
        else:
            from langchain.docstore import InMemoryDocstore
            if ((self.name in list_vectordbs(self._vectordb_dir)) & (self.vdb_dir is not None)):
                import shutil
                shutil.rmtree(self.vdb_dir)
            self._vectorstore = FAISS(embedding_function=embeddings.embedding_model, index=faiss.IndexFlatL2(embeddings.embedding_size),
                                          docstore=InMemoryDocstore({}), index_to_docstore_id={})
            self.save()
                
    def save(self) -> None:
        """Save the latest vector database.
        """
        if self.vdb_dir is not None:
            from ..utils import save_json, current_time
            self._vectorstore.save_local(self.vdb_dir)
            self.info
            self._info['last_update'] = current_time()
            save_json(self.info, os.path.join(self.vdb_dir, 'info.json'))
            if self.save_raw:
                save_json(self.data, os.path.join(self.vdb_dir, 'data.json'))
        
    @classmethod
    def from_exist(cls, name: str, embeddings: Type[BaseEmbeddingsToolkit], vectordb_dir: Optional[str] = None) -> VectorDatabase:
        """Initialise the vector database from existing files.

        Args:
            name (str): Name of the existing vector database.
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit used in this vector database.
            vectordb_dir (Optional[str], optional): Parent directory of the vector database. If None is given, the default_vectordb_dir will be used. Defaults to None.

        Returns:
            VectorDatabase: The intialised vector database.
        """
        if name not in list_vectordbs(vectordb_dir=vectordb_dir):
            raise FileNotFoundError(f'Vector database "{name}" does not exist.')
        vdb = cls(embeddings=embeddings, name=name, vectordb_dir=vectordb_dir)
        if vdb.info.get('embeddings', 'NOT_AVAIALBLE') != embeddings.name:
            if 'data.json' not in os.listdir(vdb.vdb_dir):
                raise FileNotFoundError(f'The vector database did not use the embeddings "{embeddings.name}" and no raw data was saved. Vector database cannot be loaded with this embeddings.')
            else:
                vdb._save_raw = True
                vdb._init_vectordb(embeddings=embeddings, from_exist=True)
        else:
            vdb._save_raw = 'data.json' in os.listdir(vdb.vdb_dir)
            vdb._init_vectordb(embeddings=embeddings, from_exist=True)
        return vdb
    
    @classmethod
    def from_empty(cls, embeddings: Type[BaseEmbeddingsToolkit], name: Optional[str] = None, vectordb_dir: Optional[str] = None, save_raw: bool = False) -> VectorDatabase:
        """Initialise an empty vector database.

        Args:
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkits used in the vector database.
            name (Optional[str], optional): Name of the vector database. If given, the vector database will be stored in storage. Defaults to None.
            vectordb_dir (Optional[str], optional): Parent directory of the vector database if it is not In-memory only. If None is given, the default_vectordb_dir will be used. Defaults to None.
            save_raw (bool, optional): Whether to save raw text data and metadata as a separate json file. Defaults to False.

        Returns:
            VectorDatabase: The intialised vector database.
        """
        vdb = cls(embeddings=embeddings, name=name, vectordb_dir=vectordb_dir, save_raw=save_raw)
        vdb._init_vectordb(embeddings=embeddings, from_exist=False)
        return vdb
    
    @classmethod
    def from_data(cls, index: List[str], embeddings: Type[BaseEmbeddingsToolkit],
                  data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]] = dict(), text_splitter: Type[TextSplitter] = None, name: Optional[str] = None, 
                  vectordb_dir: Optional[str] = None, save_raw: bool = False, split_text: bool = True) -> VectorDatabase:
        """Initialise the vector database with list of texts.

        Args:
            index (List[str]): List of texts to initialise the database.
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkits used in the vector database.
            data (Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]], optional): Metadata for the list of texts. Defaults to dict().
            text_splitter (Optional[Type[TextSplitter]], optional): Text splitter used to split the documents. If provided, it will be used instead of the embedding toolkit. Defaults to None.
            name (Optional[str], optional): Name of the vector database. If given, the vector database will be stored in storage. Defaults to None.
            vectordb_dir (Optional[str], optional): Parent directory of the vector database if it is not In-memory only. If None is given, the default_vectordb_dir will be used. Defaults to None.
            save_raw (bool, optional): Whether to save raw text data and metadata as a separate json file. Defaults to False.
            split_text (bool, optional): Whether to split the texts if they are too long. Defaults to True.

        Returns:
            VectorDatabase: The intialised vector database.
        """
        vdb = cls.from_empty(embeddings=embeddings, name=name, vectordb_dir=vectordb_dir, save_raw=save_raw)
        vdb.add_texts(texts=index, metadata=data, text_splitter=text_splitter, split_text=split_text)
        return vdb
    
    def add_texts(self, texts: List[str], metadata: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]] = dict(), 
                  text_splitter: Optional[Type[TextSplitter]] = None, split_text: bool = True) -> None:
        """Adding texts to the vector database.

        Args:
            texts (List[str]): List of texts to add.
            metadata (Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]], optional): Metadata for the texts. Defaults to dict().
            text_splitter (Optional[Type[TextSplitter]], optional): Text splitter used to split the documents. If provided, it will be used instead of the embedding toolkit. Defaults to None.
            split_text (bool, optional): Whether to split the texts if they are too long. Defaults to True.
        """
        docs = texts_to_documents(texts=texts, embeddings=self.embeddings, text_splitter=text_splitter, data=metadata, split_text=split_text)
        self.vectorstore.add_documents(docs)
        self.save()

    def add_documents(self, docs: List[Document], text_splitter: Optional[Type[TextSplitter]] = None, split_text: bool = True) -> None:
        """Adding documents to the vector database.

        Args:
            docs (List[Document]): List of documents to add.
            text_splitter (Optional[Type[TextSplitter]], optional): Text splitter used to split the documents. If provided, it will be used instead of the embedding toolkit. Defaults to None.
            split_text (bool, optional): Whether to split the texts if they are too long. Defaults to True.
        """
        texts = list(map(lambda x: x.page_content, docs))
        metadata = list(map(lambda x: x.metadata, docs))
        self.add_texts(texts=texts, metadata=metadata, text_splitter=text_splitter, split_text=split_text)

    def _result_dict(self, index: str, score: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to construct search results into a dictionary.

        Args:
            index (str): Index as a text string.
            score (float): Relevance score.
            metadata (Dict[str, Any]): Metadata dictionary.

        Returns:
            Dict[str, Any]:Dictionary with all information of the search result.
        """
        return dict(index=index, score=score, metadata=metadata)

    def _dictionary_filter(self, item: List[Union[str, Document]], **kwargs: Dict[str, Any]) -> bool:
        """Helper method for dictionary filtering.

        Args:
            item (List[Union[str, Document]]): Document item to check.

        Returns:
            bool: Return True if all metadata of the item satisfy the kwargs values.
        """
        for k, v in kwargs.items():
            if item[1].metadata.get(k, None) != v:
                return False
        return True

    def search(self, query: str, top_k: int = 5, index_only: bool = True, **kwargs) -> List[Union[str, Dict[str, Any]]]:
        """Similarity search of text on the vector database. Pass keyword arguments as filters on metadata.

        Args:
            query (str): Text search string query.
            top_k (int, optional): Maximum number of results to return. Defaults to 5.
            index_only (bool, optional): If set as True, only the index string will be returned. Otherwise, metadata and scores will be returned as well. Defaults to True.

        Returns:
            List[Union[str, Dict[str, Any]]]: List of search results.
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(query=query, k=top_k, fetch_k=max(top_k*2, 20), filter=kwargs)

        if index_only:
            results = list(map(lambda x: x[0].page_content, results))
        else:
            index = list(map(lambda x: x[0].page_content, results))
            metadata = list(map(lambda x: x[0].metadata, results))
            scores = list(map(lambda x: x[1], results))
            results = list(zip(index, scores, metadata))
            results = list(map(lambda x: self._result_dict(x[0], x[1], x[2]), results))
        return results
    
    def search_by_metadata(self, **kwargs: Dict[str, Any]) -> Dict[str, Document]:
        """Exact match search on metadata. Filters should be provided as key value pair arguments.

        Returns:
            Dict[str, Document]: Dictionary of saerch results, with docstore ids as the keys.
        """
        results = dict(filter(lambda x: self._dictionary_filter(x, **kwargs), self.vectorstore.docstore._dict.items()))
        return results

    def delete_by_metadata(self, **kwargs: Dict[str, Any]) -> None:
        """Remove records base on the given key value pairs criteria.

        Raises:
            ValueError: If not key value pairs given, this error will be raised. To clear the whole vector database, please use the "clear()" method.
        """
        if len(kwargs) == 0:
            raise ValueError(f'Keyword arguments must be provided to remove data from the vector database.')
        ids = list(self.search_by_metadata(**kwargs).keys())
        self.vectorstore.delete(ids)
        self.save()

    

