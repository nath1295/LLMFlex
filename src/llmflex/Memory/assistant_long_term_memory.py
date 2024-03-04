import os
from .base_memory import BaseChatMemory, list_titles, chat_memory_home
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..VectorDBs.faiss_vectordb import FaissVectorDatabase
from ..Models.Cores.base_core import BaseLLM
from ..Schemas.documents import Document
from ..Prompts.prompt_template import PromptTemplate, DEFAULT_SYSTEM_MESSAGE
from ..TextSplitters.sentence_token_text_splitter import SentenceTokenTextSplitter
from typing import List, Dict, Any, Type, Union, Tuple

class AssistantLongTermChatMemory(BaseChatMemory):

    def __init__(self, title: str, embeddings: Type[BaseEmbeddingsToolkit], text_splitter: SentenceTokenTextSplitter, from_exist: bool = True) -> None:
        self._embeddings = embeddings
        self._text_splitter = text_splitter
        super().__init__(title, from_exist)
        
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

    def get_long_term_assistant_memory(self, query: str, recent_history: Union[List[str], List[Tuple[str, str]]], 
                             llm: Type[BaseLLM], token_limit: int = 400, score_threshold: float = 0.2) -> List[str]:
        """Retriving the long term memory with the given query. Usually used together with get_token_memory.

        Args:
            query (str): Search query for the vector database. Usually the latest user input.
            recent_history (Union[List[str], List[Tuple[str, str]]]): List of interactions in the short term memory to skip in the long term memory.
            llm (Type[BaseLLM]): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens in the long term memory. Defaults to 400.
            score_threshold (float, optional): Minimum threshold for similarity score, shoulbe be between 0 to 1. Defaults to 0.2.

        Returns:
            List[str]: List of assistant chunks related to the query.
        """
        if self.interaction_count == 0:
            return []
        related = self.vectordb.search(query=query, top_k=30, index_only=False, role='assistant')
        related = list(map(lambda x: [x['metadata']['user'], x['metadata']['assistant'], x['score'], x['metadata'].get('role', None), x['index']], related))
        related = list(filter(lambda x: ((x[2] >= score_threshold) and (x[3]=='assistant')), related))
        related = list(map(lambda x: x[4], related))
        if len(related) == 0:
            print(related)
            return []
        if len(recent_history) != 0:
            if isinstance(recent_history[0], str):
                related = list(filter(lambda x: sum(list(map(lambda y: x in y, recent_history))) == 0, related))
                
            else:
                related = list(filter(lambda x: sum(list(map(lambda y: x in y[1], recent_history))) == 0, related))

        final = []
        token_count = 0
        for msg in related:
            msg_count = llm.get_num_tokens(msg)
            if (token_count + msg_count) <= token_limit:
                token_count += msg_count
                final.append(msg)
        return final

def create_long_assistant_memory_prompt(user: str, prompt_template: PromptTemplate, llm: Type[BaseLLM], memory: AssistantLongTermChatMemory,
        system: str = DEFAULT_SYSTEM_MESSAGE, short_token_limit: int = 200, long_token_limit: int = 200, score_threshold: float = 0.5) -> str:
    """Wrapper function to create full chat prompts using the prompt template given, with long term memory included in the prompt. 

    Args:
        user (str): User newest message.
        prompt_template (PromptTemplate): Prompt template to use.
        llm (Type[BaseLLM]): LLM for counting tokens.
        memory (AssistantLongTermChatMemory): The memory class with long short term functionalities.
        system (str, optional): System message for the conversation. Defaults to DEFAULT_SYSTEM_MESSAGE.
        short_token_limit (int, optional): Maximum number of tokens for short term memory. Defaults to 200.
        long_token_limit (int, optional): Maximum number of tokens for long term memory. Defaults to 200.
        score_threshold (float, optional): Minimum relevance score to be included in long term memory. Defaults to 0.5.

    Returns:
        str: The full chat prompt.
    """
    user = user.strip(' \n\r\t')
    short = memory.get_token_memory(llm=llm, token_limit=short_token_limit)
    long = memory.get_long_term_assistant_memory(query=user, recent_history=short, llm=llm, token_limit=long_token_limit, score_threshold=score_threshold)
    long = list(map(lambda x: f'"{x}"', long))
    long = '\n\n'.join(long)
    if len(long) > 0:
        system = system + '\n\n##### These are some related chunks of replies you made from previous conversations:\n' + long + \
            '\n##### No need to use the above related message chunks if they are not useful for the current conversation.'
    prompt = prompt_template.create_prompt(user=user, system=system, history=short)
    return prompt

