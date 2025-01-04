from __future__ import annotations
from .base_memory import BaseMemory
from typing import Optional, List, Dict, Any, Literal, TYPE_CHECKING
import os
if TYPE_CHECKING:
    from ..TextSplitter.base_splitter import BaseTextSplitter
    from ..Reranker.base_ranker import BaseRanker
    from ..Embeddings.Model.base_embeddings import BaseEmbeddings

VDB_TYPE = Literal['numpy', 'faiss']

class LongShortTermChatMemory(BaseMemory):
    """Long short term chat memory class."""
    def __init__(self,
            embeddings: "BaseEmbeddings",
            ranker: Optional["BaseRanker"] = None,
            text_splitter: Optional["BaseTextSplitter"] = None,
            vdb_type: VDB_TYPE = 'numpy',
            title: Optional[str] = None, 
            memory_dir: Optional[str] = None,
            **kwargs
        ) -> None:
        """Initializes the LongShortTermChatMemory class.

        Args:
            embeddings (BaseEmbeddings): The embeddings model to use for encoding and decoding.
            ranker (Optional[BaseRanker], optional): The ranker model to use for ranking the messages during search. Defaults to None.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting the longer messages. Defaults to None.
            vdb_type (VDB_TYPE, optional): The type of vector database to use. Defaults to 'numpy'.
            title (Optional[str], optional): The title of the memory. Defaults to None.
            memory_dir (Optional[str], optional): The directory to store the memory. Defaults to None.
        """
        from ..TextSplitter.sentence_splitter import SentenceTextSplitter
        from ..Reranker.huggingface_ranker import HuggingFaceRanker
        super().__init__(title, memory_dir)
        from_exist = kwargs.get('from_exist', False)
        self._vdb_dir = None if self.memory_dir is None else os.path.join(self.memory_dir, 'vdb')
        self._embeddings = embeddings
        self._ranker = ranker if ranker else HuggingFaceRanker()
        self._text_splitter = SentenceTextSplitter(tokenizer=embeddings.tokenizer, chunk_size=100, chunk_overlap=20) if text_splitter is None else text_splitter
        if vdb_type == 'numpy':
            from ..VectorDatabase.numpy_vectordb import NumpyVectorDatabase
            if from_exist:
                self._vdb = NumpyVectorDatabase.from_exist(embeddings=embeddings, vdb_dir=self._vdb_dir, text_splitter=self._text_splitter)
            else:
                self._vdb = NumpyVectorDatabase(embeddings=embeddings, text_splitter=self._text_splitter, vdb_dir=self._vdb_dir)
        elif vdb_type == 'faiss':
            from ..VectorDatabase.faiss_vectordb import FaissVectorDatabase
            if from_exist:
                self._vdb = FaissVectorDatabase.from_exist(embeddings=embeddings, vdb_dir=self._vdb_dir, text_splitter=self._text_splitter)
            else:
                self._vdb = FaissVectorDatabase(embeddings=embeddings, text_splitter=self._text_splitter, vdb_dir=self._vdb_dir)

    @classmethod
    def from_exist(cls, embeddings: "BaseEmbeddings", memory_dir: str,
            ranker: Optional["BaseRanker"] = None,
            text_splitter: Optional["BaseTextSplitter"] = None) -> LongShortTermChatMemory:
        """Initializes the LongShortTermChatMemory class from an existing memory directory.

        Args:
            embeddings (BaseEmbeddings): The embeddings model to use for encoding and decoding.
            memory_dir (str): The directory that stores the memory.
            ranker (Optional[BaseRanker], optional): The ranker model to use for ranking the messages during search. Defaults to None.
            text_splitter (Optional[BaseTextSplitter], optional): The text splitter to use for splitting the longer messages. Defaults to None.

        Returns:
            LongShortTermChatMemory: An instance of LongShortTermChatMemory initialized from the existing vector database.
        """
        if not os.path.exists(memory_dir):
            raise FileExistsError(f'Directory "{memory_dir}" does not exist.')
        elif not os.path.exists(os.path.join(memory_dir, 'info.json')):
            raise FileNotFoundError(f'"info.json" does not exist in "{memory_dir}".')
        import json
        with open(os.path.join(memory_dir, 'info.json'), 'r') as f:
            info = json.load(f)
        title = info['title']
        if cls.__name__ != info['memory_class']:
            raise Exception(f'The existing memory was not created with "{cls.__name__}" class.')
        
        vdb_info_dir = os.path.join(memory_dir, 'vdb', 'info.json')
        with open(vdb_info_dir, 'r') as f:
            vdb_info = json.load(f)
        vdb_class = vdb_info['vdb_class']
        if 'numpy' in vdb_class.lower():
            vdb_type = 'numpy'
        elif 'faiss' in vdb_class.lower():
            vdb_type = 'faiss'

        memory = cls(embeddings=embeddings, ranker=ranker, text_splitter=text_splitter, vdb_type=vdb_type, title=title, memory_dir=memory_dir, from_exist=True)
        docs = memory._vdb.data
        orders = [doc.metadata['message_index'] for doc in docs]
        history = []
        if len(orders) != 0:
            max_order = max(orders) + 1
            for i in range(max_order):
                message = [doc for doc in docs if doc.metadata['message_index'] == i][0]
                history.append(dict(role=message.metadata['role'], content=message.metadata.get('content'), tool_calls=message.metadata.get('tool_calls')))
        memory._history = history
        return memory
        

    def _update_history(self, messages: List[Dict[str, Any]]) -> None:
        """Updates the conversation history with the given messages.

        Args:
            messages (List[Dict[str, Any]]): A list of dictionaries, each representing a conversation turn.
        """
        from copy import deepcopy
        import json
        start_index = len(self.history)
        self.history.extend(messages)
        for i, msg in enumerate(messages):
            metadata = deepcopy(msg)
            metadata['message_index'] = i + start_index
            content = msg.get('content', None)
            tool_calls = msg.get('tool_calls', None)
            if content:
                if not isinstance(content, str):
                    try:
                        content = json.dumps(content)
                    except:
                        content = str(content)
                self._vdb.add_texts(texts=content, metadata=metadata)
            if tool_calls:
                for tool_call in tool_calls:
                    arguments = tool_call['function']['arguments']
                    try:
                        arg_dict = json.loads(arguments)
                    except:
                        arg_dict = arguments
                    tc_dict = deepcopy(tool_call)
                    tc_dict['function']['arguments'] = arg_dict
                    tc_str = json.dumps(tc_dict)
                    self._vdb.add_texts(texts=tc_str, metadata=metadata)

    def _remove_last_message(self) -> None:
        """Remove the latest message.
        """
        if len(self.history) > 0:
            last_index = len(self.history) - 1
            self._history = self.history[:-1]
            doc_ids = self._vdb.search_by_doc(message_index=last_index)
            if len(doc_ids) > 0:
                self._vdb.drop_ids(doc_ids)

    def get_related_messages(self, 
            query: str,
            token_limit: int = 300, 
            score_threshold: float = 0.0,
            exclude_messages: Optional[List[Dict[str, Any]]] = None
        ) -> List[Dict[str, str]]:
        """Retrieves related messages based on a query.

        Args:
            query (str): The query to search for related messages.
            token_limit (int, optional): The maximum number of tokens to consider for each message. Defaults to 300.
            score_threshold (float, optional): The minimum score required for a message to be considered related. Defaults to 0.0.
            exclude_messages (Optional[List[Dict[str, Any]]], optional): A list of messages to exclude from the search. Defaults to None.

        Returns:
            List[Dict[str, str]]: A list of related messages, each represented as a dictionary with 'role', 'content', and 'tool_calls' keys.
        """
        import json         
        def filter_fn(doc) -> bool:
            for msg in exclude_messages:
                if (doc.metadata['role'] == msg['role']):
                    if (msg.get('content') == doc.metadata.get('content')) and (msg.get('tool_calls') == doc.metadata.get('tool_calls')):
                        return False
            return True
        doc_ids = self._vdb.search_by_doc(filter_fn) if exclude_messages is not None else None
        if isinstance(doc_ids, list) and len(doc_ids) == 0:
            return []
        results = self._vdb.search(query=query, scope_ids=doc_ids, top_k=50)
        results = [res for res in results if res.score >= score_threshold]
        if len(results) == 0:
            return []
        ranked = self._ranker.rerank(query=query, docs=[res.doc for res in results])
        
        if hasattr(self._text_splitter, '_tokenizer'):
            tokenizer = self._text_splitter._tokenizer
        else:
            tokenizer = self._embeddings._tokenizer
        
        token_count = 0
        msg_list = []
        for res in ranked:
            doc = res.doc
            role = doc.metadata['role']
            res_dict = dict(role=role, content=doc.text)
            res_str = json.dumps(res_dict)
            res_token_count = tokenizer.get_num_tokens(res_str, add_special_tokens=False)
            if (token_count + res_token_count) < token_limit:
                msg_list.append(res_dict)
                token_count += res_token_count
            else:
                break
        return msg_list


        


        