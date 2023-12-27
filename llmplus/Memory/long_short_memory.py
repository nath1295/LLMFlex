import os
from .base_memory import BaseChatMemory, list_titles, chat_memory_home
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..Data.vector_database import VectorDatabase
from ..Models.Cores.base_core import BaseLLM
from ..Prompts.prompt_template import PromptTemplate, DEFAULT_SYSTEM_MESSAGE
from typing import List, Dict, Any

class LongShortTermChatMemory(BaseChatMemory):

    def __init__(self, title: str, embeddings: BaseEmbeddingsToolkit, from_exist: bool = True) -> None:
        self._embeddings = embeddings
        super().__init__(title, from_exist)
        
    @property
    def embeddings(self) -> BaseEmbeddingsToolkit:
        """Embeddings toolkit.

        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit.
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

    def save_interaction(self, user_input: str, assistant_output: str) -> None:
        """Saving an interaction to the memory.

        Args:
            user_input (str): User input.
            assistant_output (str): Chatbot output.
        """
        user_input = user_input.strip(' \n\r\t')
        assistant_output = assistant_output.strip(' \n\r\t')
        self.vectordb.add_texts(texts=[f'{user_input}\n\n{assistant_output}'],
                                metadata=dict(user=user_input, assistant=assistant_output, order=self.interaction_count))

    def remove_last_interaction(self) -> None:
        """Remove the latest interaction.
        """
        if self.interaction_count != 0:
            self.vectordb.delete_by_metadata(order=self.interaction_count-1)

    def get_long_term_memory(self, query: str, short_term_memory: List[List[str]], 
                             llm: BaseLLM, token_limit: int = 400, score_threshold: float = 0.2) -> List[List[str]]:
        """Retriving the long term memory with the given query. Usually used together with get_token_memory.

        Args:
            query (str): Search query for the vector database. Usually the latest user input.
            short_term_memory (List[List[str]]): List of interactions in the short term memory to skip in the long term memory.
            llm (BaseLLM): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens in the long term memory. Defaults to 400.
            score_threshold (float, optional): Minimum threshold for similarity score, shoulbe be between 0 to 1. Defaults to 0.2.

        Returns:
            List[List[str]]: List of interactions related to the query.
        """
        if self.interaction_count == 0:
            return []
        related = self.vectordb.search(query=query, top_k=10, index_only=False)
        related = list(map(lambda x: [x['metadata']['user'], x['metadata']['assistant'], x['score']], related))
        related = list(filter(lambda x: x[2] >= score_threshold, related))
        if len(related) == 0:
            return []
        long_concat = list(map(lambda x: f'Input: {x[0]}\nOutput: {x[1]}', related))
        short_conccat = list(map(lambda x: f'Input: {x[0]}\nOutput: {x[1]}', short_term_memory))
        is_exist = list(map(lambda x: x in short_conccat, long_concat))

        final = []
        token_count = 0
        for i, msg in enumerate(related):
            if not is_exist[i]:
                msg_count = llm.get_num_tokens(msg[0]) + llm.get_num_tokens(msg[1])
                if (token_count + msg_count) <= token_limit:
                    token_count += msg_count
                    final.append(msg[:2])
        return final

def create_long_short_prompt(user: str, prompt_template: PromptTemplate, llm: BaseLLM, memory: LongShortTermChatMemory,
        system: str = DEFAULT_SYSTEM_MESSAGE, short_token_limit: int = 200, long_token_limit: int = 200, score_threshold: float = 0.5) -> str:
    """Wrapper function to create full chat prompts using the prompt template given, with long term memory included in the prompt. 

    Args:
        user (str): User newest message.
        prompt_template (PromptTemplate): Prompt template to use.
        llm (BaseLLM): LLM for counting tokens.
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
    long = memory.get_long_term_memory(query=user, short_term_memory=short, llm=llm, token_limit=long_token_limit, score_threshold=score_threshold)
    if len(long) > 0:
        system = system + '\n\n##### These are some related message from previous conversations:\n' + prompt_template.format_history(long) + \
            '\n##### No need to use the above related messages if they are not useful for the current conversation.'
    prompt = prompt_template.create_chat_prompt(user=user, system=system, history=short)
    return prompt

