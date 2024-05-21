import os
from .base_memory import BaseChatMemory, list_titles, chat_memory_home
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..VectorDBs.faiss_vectordb import FaissVectorDatabase
from ..Models.Cores.base_core import BaseLLM
from ..Schemas.documents import Document
from ..Rankers.base_ranker import BaseRanker
from ..TextSplitters.base_text_splitter import BaseTextSplitter
from ..Prompts.prompt_template import PromptTemplate
from ..KnowledgeBase.knowledge_base import KnowledgeBase
from ..TextSplitters.sentence_token_text_splitter import SentenceTokenTextSplitter
from typing import List, Dict, Any, Type, Union, Tuple, Optional

class LongShortTermChatMemory(BaseChatMemory):

    def __init__(self, title: str, 
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
            title (str): Title of the chat.
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit for the vector database for storing chat history.
            llm (Optional[BaseLLM]): LLM for counting tokens.
            ranker (Optional[BaseRanker], optional): Reranker for long term memory retrieval. Defaults to None.
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
        super().__init__(title=title, from_exist=from_exist, system=system)
        
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
        if ((from_exist) & (self.title in list_titles())):
            self._vectordb = FaissVectorDatabase.from_exist(name=os.path.basename(self.chat_dir), 
                                                       embeddings=self.embeddings, vectordb_dir=chat_memory_home())

        else:
            self._vectordb = FaissVectorDatabase.from_documents(embeddings=self.embeddings,
                                                    docs=[],
                                                    name=os.path.basename(self.chat_dir), 
                                                    vectordb_dir=chat_memory_home())
            self.vectordb._info['title'] = self.title
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

    def get_long_term_memory(self, query: str, recent_history: Union[List[str], List[Tuple[str, str]]], 
                             llm: Type[BaseLLM], token_limit: int = 400, similarity_score_threshold: float = 0.2, relevance_score_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Retriving the long term memory with the given query. Usually used together with get_token_memory.

        Args:
            query (str): Search query for the vector database. Usually the latest user input.
            recent_history (Union[List[str], List[Tuple[str, str]]]): List of interactions in the short term memory to skip in the long term memory.
            llm (Type[BaseLLM]): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens in the long term memory. Defaults to 400.
            similarity_score_threshold (float, optional): Minimum threshold for similarity score, shoulbe be between 0 to 1. Defaults to 0.2.
            relevance_score_threshold (float, optional): Minimum threshold for relevance score for the reranker, shoulbe be between 0 to 1. Defaults to 0.8.

        Returns:
            List[Dict[str, Any]]: List of chunks related to the query and their respective speaker.
        """
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

    def create_prompt_with_memory(self, user: str, prompt_template: PromptTemplate, llm: Type[BaseLLM],
            system: Optional[str] = None, recent_token_limit: int = 200, 
            knowledge_base: Optional[KnowledgeBase] = None, relevance_token_limit: int = 200, relevance_score_threshold: float = 0.8, 
            similarity_score_threshold: float = 0.5, **kwargs) -> str:
        """Wrapper function to create full chat prompts using the prompt template given, with long term memory included in the prompt. 

        Args:
            user (str): User newest message.
            prompt_template (PromptTemplate): Prompt template to use.
            llm (Type[BaseLLM]): LLM for counting tokens.
            system (Optional[str], optional): System message to override the default system message for the memory. Defaults to None.
            recent_token_limit (int, optional): Maximum number of tokens for recent term memory. Defaults to 200.
            knowledge_base (Optional[KnowledgeBase]): Knowledge base that helps the assistant to answer questions. Defaults to None.
            relevance_token_limit (int, optional): Maximum number of tokens for search results from the knowledge base if a knowledge base is given. Defaults to 200.
            relevance_score_threshold (float, optional): Reranking score threshold for knowledge base search if a knowledge base is given. Defaults to 0.8.
            similarity_score_threshold (float, optional): Long term memory similarity score threshold. Defaults to 0.5.


        Returns:
            str: The full chat prompt.
        """
        user = user.strip()
        short = self.get_token_memory(llm=llm, token_limit=recent_token_limit)
        system = self.system if not isinstance(system, str) else system
        messages: list = [dict(role='system', content=self.system)] + prompt_template.format_history(short, return_list=True)

        # Gathering knowledge
        results = self.get_long_term_memory(query=user, recent_history=short, llm=llm, 
                        token_limit=relevance_token_limit, similarity_score_threshold=similarity_score_threshold, relevance_score_threshold=relevance_score_threshold)
        results = list(map(lambda x: dict(text=x['chunk'], role=x['role'], relevance_score=x['relevance_score'], info_type='memory'), results))

        if knowledge_base is not None:
            kb_res = knowledge_base.search(query=user, token_limit=relevance_token_limit, fetch_k=50, count_fn=llm.get_num_tokens, relevance_score_threshold=relevance_score_threshold)
            kb_res = list(map(lambda x: dict(text=x.index, relevance_score=x.rank_score, info_type='kb'), kb_res))
            results.extend(kb_res)

        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # filter by token counts
        res = []
        token_count = 0
        for r in results:
            rcount = llm.get_num_tokens(r['text'])
            if token_count + rcount <= relevance_token_limit:
                res.append(r)
                token_count += rcount
            else:
                break

        # formatting information
        if len(res) != 0:
            import json
            as_mem = list(filter(lambda x: res.get('role') == 'assistant', res))
            us_mem = list(filter(lambda x: res.get('role') == 'user', res))
            kb_mem = list(filter(lambda x: res.get('role') == 'user', res))
            if prompt_template.allow_custom_role:
                res = json.dumps({'Information from knowledge base': res}, indent=4)
                messages.extend([dict(role='user', content=user), dict(role='extra information', content=res)])
                prompt = prompt_template.create_custom_prompt(messages=messages, add_generation_prompt=True) 
            else:
                messages.append(dict(role='user', content=user))
                prompt = prompt_template.create_custom_prompt(messages=messages, add_generation_prompt=True)
                res = json.dumps({'Information from knowledge base': res}, indent=4)
                prompt += f'{res}\n\nResponse: '
        else:
            messages.append(dict(role='user', content=user))
            prompt = prompt_template.create_custom_prompt(messages=messages, add_generation_prompt=True)    

        return prompt

