import os
from .base_memory import BaseChatMemory, list_chat_ids, chat_memory_home
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..VectorDBs.faiss_vectordb import FaissVectorDatabase
from ..Models.Cores.base_core import BaseLLM
from ..Schemas.documents import Document
from ..Rankers.base_ranker import BaseRanker
from ..TextSplitters.base_text_splitter import BaseTextSplitter
from ..TextSplitters.sentence_token_text_splitter import SentenceTokenTextSplitter
from typing import List, Dict, Any, Type, Union, Tuple, Optional

class LongShortTermChatMemory(BaseChatMemory):

    def __init__(self, chat_id: str, 
                 embeddings: Type[BaseEmbeddingsToolkit], 
                 llm: Optional[BaseLLM], 
                 ranker: Optional[BaseRanker] = None,
                 text_splitter: Optional[BaseTextSplitter] = None,
                 ts_lang_model: str = 'en_core_web_sm',
                 chunk_size: int = 400,
                 chunk_overlap: int = 40,
                 from_exist: bool = True,
                 system: Optional[str] = None) -> None:
        """Initialising the memory class.

        Args:
            chat_id (str): Chat ID.
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit for the vector database for storing chat history.
            llm (Optional[BaseLLM]): LLM for counting tokens.
            ranker (Optional[BaseRanker], optional): Reranker for long term memory retrieval. Defaults to None.
            text_splitter (Optional[BaseTextSplitter], optional): Text splitter to use. If None given, it will be created with other arguments. Defaults to None.
            ts_lang_model (str, optional): Language model for the sentence text splitter. Defaults to 'en_core_web_sm'.
            chunk_size (int, optional): Chunk size for the text splitter. Defaults to 400.
            chunk_overlap (int, optional): Chunk overlap for the text splitter. Defaults to 40.
            from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.
            system (Optional[str], optional): System message for the chat. If None is given, the default system message or the stored system message will be used. Defaults to None.
        """
        self._embeddings = embeddings
        count_token_fn = self.embeddings.tokenizer.get_num_tokens if llm is None else llm.get_num_tokens
        self._text_splitter = SentenceTokenTextSplitter(
            count_token_fn=count_token_fn, 
            language_model=ts_lang_model, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        ) if text_splitter is None else text_splitter
        if ranker is None:
            from ..Rankers.flashrank_ranker import FlashrankRanker
            self._ranker = FlashrankRanker()
        else:
            from ..utils import validate_type
            self._ranker = validate_type(ranker, BaseRanker)
        super().__init__(chat_id=chat_id, from_exist=from_exist, system=system)
        
    @property
    def embeddings(self) -> BaseEmbeddingsToolkit:
        """Embeddings toolkit.

        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit.
        """
        return self._embeddings

    @property
    def vectordb(self) -> FaissVectorDatabase:
        """Vector database for saving the chat history.

        Returns:
            FaissVectorDatabase: Vector database for saving the chat history.
        """
        return self._vectordb

    @property
    def info(self) -> Dict[str, Any]:
        """Information of the chat.

        Returns:
            Dict[str, Any]: Information of the chat.
        """
        return self.vectordb.info

    @property
    def text_splitter(self) -> SentenceTokenTextSplitter:
        """Sentence text splitter.

        Returns:
            SentenceTokenTextSplitter: Sentence text splitter.
        """
        return self._text_splitter
    
    @property
    def ranker(self) -> BaseRanker:
        """Reranker.

        Returns:
            BaseRanker: Reranker.
        """
        return self._ranker

    @property
    def _data(self) -> Dict[str, Document]:
        """Raw data from the vector database.

        Returns:
            Dict[str, Document]: Raw data from the vector database.
        """
        return self.vectordb.data


    def _init_memory(self, from_exist: bool = True) -> None:
        """Method to initialise the components in the memory.

        Args:
            from_exist (bool, optional): Whether to initialise from existing files. Defaults to True.
        """
        if ((from_exist) & (self.chat_id in list_chat_ids())):
            self._vectordb = FaissVectorDatabase.from_exist(name=os.path.basename(self.chat_dir), 
                                                       embeddings=self.embeddings, vectordb_dir=chat_memory_home())
            self._title = self.vectordb.info.get('title', 'New Chat')

        else:
            self._vectordb = FaissVectorDatabase.from_documents(embeddings=self.embeddings,
                                                    docs=[],
                                                    name=os.path.basename(self.chat_dir), 
                                                    vectordb_dir=chat_memory_home())
            self.info['title'] = self.title
            self.save()

    def save(self) -> None:
        """Save the current state of the memory.
        """
        self.vectordb.save()

    def save_interaction(self, user_input: str, assistant_output: str, **kwargs) -> None:
        """Saving an interaction to the memory.

        Args:
            user_input (str): User input.
            assistant_output (str): Chatbot output.
        """
        from copy import deepcopy
        user_input = user_input.strip(' \n\r\t')
        assistant_output = assistant_output.strip(' \n\r\t')
        user_chunks = self.text_splitter.split_text(user_input)
        assistant_chunks = self.text_splitter.split_text(assistant_output)
        metadata = dict(user=user_input, assistant=assistant_output, order=self.interaction_count)
        for k, v in kwargs.items():
            if k not in ['user', 'assistant', 'order',  'role']:
                metadata[k] = v
        metadata_user = deepcopy(metadata)
        metadata_user['role'] = 'user'
        metadata_assistant = deepcopy(metadata)
        metadata_assistant['role'] = 'assistant'
        self.vectordb.add_texts(texts=user_chunks, metadata=metadata_user, split_text=False)
        self.vectordb.add_texts(texts=assistant_chunks, metadata=metadata_assistant, split_text=False)

    def remove_last_interaction(self) -> None:
        """Remove the latest interaction.
        """
        if self.interaction_count != 0:
            self.vectordb.delete_by_metadata(order=self.interaction_count-1)

    def get_long_term_memory(self, query: str,llm: Type[BaseLLM], recent_history: Optional[Union[List[str], List[Tuple[str, str]]]] = None, 
                             token_limit: int = 400, similarity_score_threshold: float = 0.2, relevance_score_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Retriving the long term memory with the given query. Usually used together with get_token_memory.

        Args:
            query (str): Search query for the vector database. Usually the latest user input.
            llm (Type[BaseLLM]): LLM to count tokens.
            recent_history (Optional[Union[List[str], List[Tuple[str, str]]]], optional): List of interactions in the short term memory to skip in the long term memory. Defaults to None.
            token_limit (int, optional): Maximum number of tokens in the long term memory. Defaults to 400.
            similarity_score_threshold (float, optional): Minimum threshold for similarity score, shoulbe be between 0 to 1. Defaults to 0.2.
            relevance_score_threshold (float, optional): Minimum threshold for relevance score for the reranker, shoulbe be between 0 to 1. Defaults to 0.8.

        Returns:
            List[Dict[str, Any]]: List of chunks related to the query and their respective speaker.
        """
        recent_history = [] if recent_history is None else recent_history
        if self.interaction_count == 0:
            return []
        related = self.vectordb.search(query=query, top_k=30, index_only=False)
        related = list(filter(lambda x: (x['score'] >= similarity_score_threshold), related))
        if len(related) == 0:
            return []
        if len(recent_history) != 0:
            if isinstance(recent_history[0], str):
                related = filter(lambda x: all(x['metadata']['user'] not in c for c in recent_history) 
                                 and all(x['metadata']['assistant'] not in c for c in recent_history), related)
                
            else:
                related = filter(lambda x: all(((x['metadata']['user'] not in c[0]) and (x['metadata']['assistant'] not in c[1])) for c in recent_history), related)
        related = list(related)
        if len(related) == 0:
            return []
        
        related = self.ranker.rerank(query=query, elements=related, top_k=len(related))
        related = filter(lambda x: x.rank_score >= relevance_score_threshold, related)

        final = []
        token_count = 0
        for msg in related:
            msg_count = llm.get_num_tokens(msg.index)
            if (token_count + msg_count) <= token_limit:
                token_count += msg_count
                final.append(dict(role=msg.metadata.get('role', 'unknown'), chunk=msg.index, relevance_score=msg.rank_score))
        return final
