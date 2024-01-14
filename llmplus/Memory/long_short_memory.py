import os
from .base_memory import BaseChatMemory, list_titles, chat_memory_home
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..Data.vector_database import VectorDatabase
from ..Models.Cores.base_core import BaseLLM
from ..Prompts.prompt_template import PromptTemplate, DEFAULT_SYSTEM_MESSAGE
from typing import List, Dict, Any, Type, Union, Tuple

class LongShortTermChatMemory(BaseChatMemory):

    def __init__(self, title: str, embeddings: Type[BaseEmbeddingsToolkit], from_exist: bool = True) -> None:
        self._embeddings = embeddings
        super().__init__(title, from_exist)
        
    @property
    def embeddings(self) -> Type[BaseEmbeddingsToolkit]:
        """Embeddings toolkit.

        Returns:
            Type[BaseEmbeddingsToolkit]: Embeddings toolkit.
        """
        return self._embeddings

    @property
    def vectordb(self) -> VectorDatabase:
        """Vector database for saving the chat history.

        Returns:
            VectorDatabase: Vector database for saving the chat history.
        """
        return self._vectordb

    @property
    def _data(self) -> List[Dict[str, Any]]:
        """Raw data from the vector database.

        Returns:
            List[Dict[str, Any]]: Raw data from the vector database.
        """
        return self.vectordb.data

    def _init_memory(self, from_exist: bool = True) -> None:
        """Method to initialise the components in the memory.

        Args:
            from_exist (bool, optional): Whether to initialise from existing files. Defaults to True.
        """
        if ((from_exist) & (self.title in list_titles())):
            self._vectordb = VectorDatabase.from_exist(name=os.path.basename(self.chat_dir), 
                                                       embeddings=self.embeddings, vectordb_dir=chat_memory_home())

        else:
            self._vectordb = VectorDatabase.from_empty(embeddings=self.embeddings, 
                                                       name=os.path.basename(self.chat_dir), 
                                                       vectordb_dir=chat_memory_home(), save_raw=True)
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
        user_input = user_input.strip(' \n\r\t')
        assistant_output = assistant_output.strip(' \n\r\t')
        metadata = dict(user=user_input, assistant=assistant_output, order=self.interaction_count)
        for k, v in kwargs.items():
            if k not in ['user', 'assistant', 'order']:
                metadata[k] = v
        self.vectordb.add_texts(texts=[f'{user_input}\n\n{assistant_output}'],
                                metadata=metadata)

    def remove_last_interaction(self) -> None:
        """Remove the latest interaction.
        """
        if self.interaction_count != 0:
            self.vectordb.delete_by_metadata(order=self.interaction_count-1)

    def get_long_term_memory(self, query: str, recent_history: Union[List[str], List[Tuple[str, str]]], 
                             llm: Type[BaseLLM], token_limit: int = 400, score_threshold: float = 0.2) -> List[Tuple[str, str]]:
        """Retriving the long term memory with the given query. Usually used together with get_token_memory.

        Args:
            query (str): Search query for the vector database. Usually the latest user input.
            recent_history (Union[List[str], List[Tuple[str, str]]]): List of interactions in the short term memory to skip in the long term memory.
            llm (Type[BaseLLM]): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens in the long term memory. Defaults to 400.
            score_threshold (float, optional): Minimum threshold for similarity score, shoulbe be between 0 to 1. Defaults to 0.2.

        Returns:
            List[Tuple[str, str]]: List of interactions related to the query.
        """
        if self.interaction_count == 0:
            return []
        related = self.vectordb.search(query=query, top_k=15, index_only=False)
        related = list(map(lambda x: [x['metadata']['user'], x['metadata']['assistant'], x['score']], related))
        related = list(filter(lambda x: x[2] >= score_threshold, related))
        related = list(map(lambda x: tuple(x[:2]), related))
        if len(related) == 0:
            return []
        if len(recent_history) != 0:
            if isinstance(recent_history[0], str):
                length = len(recent_history)
                is_even = length % 2 == 0
                lead = recent_history[0]
                alt_history = recent_history if is_even else recent_history[1:]
                alt_history = list(map(lambda x: (alt_history[x * 2], alt_history[x * 2 + 1]), range(len(alt_history) // 2)))
                related = list(filter(lambda x: x not in alt_history, related))
                if not is_even:
                    related = list(filter(lambda x: x[1] != lead, related))
                
            else:
                alt_history = list(map(lambda x: tuple(x), recent_history))
                related = list(filter(lambda x: x not in alt_history, related))

        final = []
        token_count = 0
        for msg in related:
            msg_count = llm.get_num_tokens(msg[0]) + llm.get_num_tokens(msg[1])
            if (token_count + msg_count) <= token_limit:
                token_count += msg_count
                final.append(msg)
        return final

def create_long_short_prompt(user: str, prompt_template: PromptTemplate, llm: Type[BaseLLM], memory: LongShortTermChatMemory,
        system: str = DEFAULT_SYSTEM_MESSAGE, short_token_limit: int = 200, long_token_limit: int = 200, score_threshold: float = 0.5) -> str:
    """Wrapper function to create full chat prompts using the prompt template given, with long term memory included in the prompt. 

    Args:
        user (str): User newest message.
        prompt_template (PromptTemplate): Prompt template to use.
        llm (Type[BaseLLM]): LLM for counting tokens.
        memory (LongShortTermChatMemory): The memory class with long short term functionalities.
        system (str, optional): System message for the conversation. Defaults to DEFAULT_SYSTEM_MESSAGE.
        short_token_limit (int, optional): Maximum number of tokens for short term memory. Defaults to 200.
        long_token_limit (int, optional): Maximum number of tokens for long term memory. Defaults to 200.
        score_threshold (float, optional): Minimum relevance score to be included in long term memory. Defaults to 0.5.

    Returns:
        str: The full chat prompt.
    """    """"""    
    user = user.strip(' \n\r\t')
    short = memory.get_token_memory(llm=llm, token_limit=short_token_limit)
    long = memory.get_long_term_memory(query=user, recent_history=short, llm=llm, token_limit=long_token_limit, score_threshold=score_threshold)
    if len(long) > 0:
        system = system + '\n\n##### These are some related message from previous conversations:\n' + prompt_template.format_history(long) + \
            '\n##### No need to use the above related messages if they are not useful for the current conversation.'
    prompt = prompt_template.create_prompt(user=user, system=system, history=short)
    return prompt

