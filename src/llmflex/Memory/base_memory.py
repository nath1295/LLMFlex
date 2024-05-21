import os
from ..utils import get_config
from ..Models.Cores.base_core import BaseLLM
from ..Prompts.prompt_template import DEFAULT_SYSTEM_MESSAGE, PromptTemplate
from ..KnowledgeBase.knowledge_base import KnowledgeBase
from typing import List, Dict, Any, Type, Tuple, Optional


def chat_memory_home() -> str:
    """Return the default directory for saving chat memories.

    Returns:
        str: The default directory for saving chat memories.
    """
    history_dir = os.path.join(get_config()['package_home'], 'chat_memories')
    os.makedirs(history_dir, exist_ok=True)
    return history_dir

def list_chat_dirs() -> List[str]:
    """List the directories of all chat memories.

    Returns:
        List[str]: List of directories of all chat memories.
    """
    chats_dir = chat_memory_home()
    dirs = list(map(lambda x: os.path.join(chats_dir, x), os.listdir(chats_dir)))
    dirs = list(filter(lambda x: os.path.basename(x).startswith('chat_'), dirs))
    dirs = list(filter(lambda x: ((os.path.isdir(x)) & ('info.json' in os.listdir(x))), dirs))
    return dirs

def title_dir_map() -> Dict[str, str]:
    """Return a dictionary with chat titles as keys and their respective directories as values.

    Returns:
        Dict[str, str]: A dictionary with chat titles as keys and their respective directories as values.
    """
    from ..utils import read_json
    dirs = list_chat_dirs()
    titles = list(map(lambda x: read_json(os.path.join(x, 'info.json'))['title'], dirs))
    return dict(zip(titles, dirs))

def list_titles() -> List[str]:
    """Return a list of chat titles.

    Returns:
        List[str]: List of chat titles, sorted by last update descendingly.
    """
    from ..utils import read_json
    mapper = list(title_dir_map().items())
    mapper = list(map(lambda x: (x[0], read_json(os.path.join(x[1], 'info.json'))['last_update']), mapper))
    mapper.sort(key=lambda x: x[1], reverse=True)
    return list(map(lambda x: x[0], mapper))   

class BaseChatMemory:
    """Base class for chat memory.
    """
    def __init__(self, title: str, from_exist: bool = True, system: Optional[str] = None) -> None:
        """Initialising the memory class.

        Args:
            title (str): Title of the chat.
            from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.
            system (Optional[str], optional): System message for the chat. If None is given, the default system message or the stored system message will be used. Defaults to None.
        """
        title = title.strip(' \n\r\t')
        if title == '':
            raise ValueError('Chat title cannot be an empty string.')
        self._title = title
        self._init_memory(from_exist=from_exist)
        self.info['system'] = system.strip() if system is not None else self.info.get('system', DEFAULT_SYSTEM_MESSAGE)
        self.save()

    @property
    def title(self) -> str:
        """Chat title.

        Returns:
            str: Chat title.
        """
        return self._title
    
    @property
    def chat_dir(self) -> str:
        """Directory of the chat.

        Returns:
            str: Directory of the chat.
        """
        mapper = title_dir_map()
        if hasattr(self, '_chat_dir'):
            return self._chat_dir
        elif self.title in list(mapper.keys()):
            self._chat_dir = mapper[self.title]
            return self._chat_dir
        elif len(list(mapper.keys())) == 0:
            self._chat_dir = os.path.join(chat_memory_home(), 'chat_0')
            os.makedirs(self._chat_dir, exist_ok=True)
            return self._chat_dir
        else:
            dirs = list(mapper.values())
            new = max(list(map(lambda x: int(os.path.basename(x).removeprefix('chat_')), dirs))) + 1
            self._chat_dir = os.path.join(chat_memory_home(), f'chat_{new}')
            os.makedirs(self._chat_dir, exist_ok=True)
            return self._chat_dir
    
    @property
    def info(self) -> Dict[str, Any]:
        """Information of the chat.

        Returns:
            Dict[str, Any]: Information of the chat.
        """
        if hasattr(self, '_info'):
            return self._info
        elif 'info.json' in os.listdir(self.chat_dir):
            from ..utils import read_json
            self._info = read_json(os.path.join(self.chat_dir, 'info.json'))
            return self._info
        else:
            from ..utils import save_json, current_time
            self._info = dict(title=self.title, last_update=current_time())
            save_json(self._info, os.path.join(self.chat_dir, 'info.json'))
            return self._info

    @property
    def system(self) -> str:
        """Default system message of the memory.

        Returns:
            str: Default system message of the memory.
        """
        return self.info.get('system', DEFAULT_SYSTEM_MESSAGE)

    @property    
    def history(self) -> List[Tuple[str, str]]:
        """Entire chat history.

        Returns:
            List[Tuple[str, str]]: Entire chat history.
        """
        history = list(map(lambda x: [x.metadata['user'], x.metadata['assistant'], x.metadata['order']], self._data.values()))
        if len(history) == 0:
            return []
        count = max(list(map(lambda x: x[2], history))) + 1
        history = list(map(lambda x: list(filter(lambda y: y[2] == x, history))[0], range(count)))
        history.sort(key=lambda x: x[2], reverse=False)
        return list(map(lambda x: tuple(x[:2]), history))

    @property
    def interaction_count(self) -> int:
        """Number of interactions.

        Returns:
            int: Number of interactions.
        """
        return len(self.history)

    def _init_memory(self, from_exist: bool = True) -> None:
        """Method to initialise the components in the memory.

        Args:
            from_exist (bool, optional): Whether to initialise from existing files. Defaults to True.
        """
        if ((from_exist) & (self.title in list_titles())):
            import pickle
            with open(os.path.join(self.chat_dir, 'data.pkl'), 'rb') as f:
                self._data = pickle.load(f)

        else:
            self._data = dict()
            self.save()

    def save(self) -> None:
        """Save the current state of the memory.
        """
        from ..utils import save_json, current_time
        import pickle
        self.info
        self._info['last_update'] = current_time()
        save_json(self._info, os.path.join(self.chat_dir, 'info.json'))
        with open(os.path.join(self.chat_dir, 'data.pkl'), 'wb') as f:
            pickle.dump(self._data, f)

    def update_system_message(self, system: str) -> None:
        """Update the default system message for the memory.

        Args:
            system (str): New system message.
        """
        self.info['system'] = system.strip()
        self.save()

    def save_interaction(self, user_input: str, assistant_output: str, **kwargs) -> None:
        """Saving an interaction to the memory.

        Args:
            user_input (str): User input.
            assistant_output (str): Chatbot output.
        """
        from ..Schemas.documents import Document
        from copy import deepcopy
        user_input = user_input.strip(' \n\r\t')
        assistant_output = assistant_output.strip(' \n\r\t')
        metadata = dict(user=user_input, assistant=assistant_output, order=self.interaction_count)
        for k, v in kwargs.items():
            if k not in ['user', 'assistant', 'order', 'role']:
                metadata[k] = v
        new_key = max(list(self._data.keys())) + 1 if len(self._data) != 0 else 0
        meta_user = deepcopy(metadata)
        meta_user['role'] = 'user'
        meta_assistant = deepcopy(metadata)
        meta_assistant['role'] = 'assistant'
        self._data[new_key] = Document(
            index=user_input,
            metadata=meta_user
        )
        self._data[new_key + 1] = Document(
            index=assistant_output,
            metadata=meta_assistant
        )
        self.save()

    def remove_last_interaction(self) -> None:
        """Remove the latest interaction.
        """
        if len(self._data) != 0:
            order = self.interaction_count
            data = list(filter(lambda x: x.metadata['order'] != order, list(self._data.values())))
            self._data = dict(zip(range(len(data)), data))
            self.save()

    def clear(self) -> None:
        """Empty the whole chat history.
        """
        self._init_memory(from_exist=False)

    def get_recent_memory(self, k: int = 3) -> List[Tuple[str, str]]:
        """Get the last k interactions as a list.

        Args:
            k (int, optional): Maximum number of latest interactions. Defaults to 3.

        Returns:
            List[Tuple[str, str]]: List of interactions.
        """
        from copy import deepcopy
        history = self.history
        results = history if len(history) <= k else history[-k:]
        return deepcopy(results)

    def get_token_memory(self, llm: Type[BaseLLM], token_limit: int = 400) -> List[str]:
        """Get the latest conversation limited by number of tokens.

        Args:
            llm (Type[BaseLLM]): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens allowed. Defaults to 400.

        Returns:
            List[str]: List of most recent messages.
        """
        if len(self.history) == 0:
            return []
        tk_count = 0
        history = sum(list(map(list, self.history)), [])
        history = list(reversed(history))
        results = list()
        for m in history:
            msg_count = llm.get_num_tokens(m)
            if (((msg_count + tk_count) <= token_limit) | (len(results) < 2)):
                results = [m] + results
                tk_count += msg_count
            else:
                break
        return results
    
    def create_prompt_with_memory(self, user: str, prompt_template: PromptTemplate, llm: Type[BaseLLM],
            system: Optional[str] = None, recent_token_limit: int = 200, 
            knowledge_base: Optional[KnowledgeBase] = None, relevance_token_limit: int = 200, relevance_score_threshold: float = 0.8, **kwargs) -> str:
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


        Returns:
            str: The full chat prompt.
        """
        user = user.strip()
        short = self.get_token_memory(llm=llm, token_limit=recent_token_limit)
        system = self.system if not isinstance(system, str) else system
        messages: list = [dict(role='system', content=self.system)] + prompt_template.format_history(short, return_list=True)
        if knowledge_base is not None:
            res = knowledge_base.search(query=user, token_limit=relevance_token_limit, fetch_k=50, count_fn=llm.get_num_tokens, relevance_score_threshold=relevance_score_threshold)
            if len(res) != 0:
                import json
                res = list(map(lambda x: x.index, res))
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
        else:
            messages.append(dict(role='user', content=user))
            prompt = prompt_template.create_custom_prompt(messages=messages, add_generation_prompt=True)
        return prompt





