import os
from ..utils import get_config
from ..Models.Cores.base_core import BaseLLM
from ..Prompts.prompt_template import DEFAULT_SYSTEM_MESSAGE
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
    import re
    re_chat = re.compile(r'chat_\d+')
    chats_dir = chat_memory_home()
    dirs = list(map(lambda x: os.path.join(chats_dir, x), os.listdir(chats_dir)))
    dirs = list(filter(lambda x: re_chat.match(os.path.basename(x)), dirs))
    dirs = list(filter(lambda x: ((os.path.isdir(x)) & ('info.json' in os.listdir(x))), dirs))
    return dirs

def list_chat_ids() -> List[str]:
    """Return a list of existing chat ids.

    Returns:
        List[str]: Return a list of existing chat ids, sorted by last update descendingly.
    """
    from ..utils import read_json
    chat_dirs = list_chat_dirs()
    chat_infos = list(map(lambda x: [read_json(os.path.join(x, 'info.json')), x], chat_dirs))
    chat_infos.sort(key=lambda x: x[0]['last_update'], reverse=True)
    chat_ids = list(map(lambda x: os.path.basename(x[1]), chat_infos))
    return chat_ids

def list_titles() -> List[str]:
    """Return a list of chat titles.

    Returns:
        List[str]: List of chat titles, sorted by last update descendingly.
    """
    from ..utils import read_json
    chat_ids = list_chat_ids()
    home = chat_memory_home()
    titles = list(map(lambda x: read_json(os.path.join(home, x, 'info.json'))['title'], chat_ids))
    return titles   

def get_new_chat_id() -> str:
    """Get an unused chat id.

    Returns:
        str: New chat id.
    """
    chat_ids = list_chat_ids()
    if len(chat_ids) == 0:
        return 'chat_0'
    indexes = list(map(lambda x: int(x.removeprefix('chat_')), chat_ids))
    max_index = max(indexes)
    for i in range(max_index + 1):
        if f'chat_{i}' not in chat_ids:
            return f'chat_{i}'
    return f'chat_{max_index + 1}'
    
def get_title_from_id(chat_id: str) -> str:
    """Getting the title from Chat ID.

    Args:
        chat_id (str): Chat ID.

    Returns:
        str: Title of the memory.
    """
    if chat_id not in list_chat_ids():
        raise FileNotFoundError(f'Chat ID "{chat_id}" does not exist.')
    from ..utils import read_json
    return read_json(os.path.join(chat_memory_home(), chat_id, 'info.json'))['title']

def get_dir_from_id(chat_id: str) -> str:
    """Geet the memory directory given the chat ID.

    Args:
        chat_id (str): Chat ID.

    Returns:
        str: Memory directory.
    """
    return os.path.join(chat_memory_home(), chat_id)

class BaseChatMemory:
    """Base class for chat memory.
    """
    def __init__(self, chat_id: str, from_exist: bool = True, system: Optional[str] = None) -> None:
        """Initialising the memory class.

        Args:
            chat_id (str): Chat ID.
            from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.
            system (Optional[str], optional): System message for the chat. If None is given, the default system message or the stored system message will be used. Defaults to None.
        """
        import re
        re_chat = re.compile(r'chat_\d+')
        if not re_chat.match(chat_id):
            raise ValueError('Invalid chat ID.')
        self._chat_id = chat_id
        self._init_memory(from_exist=from_exist)
        self.info['system'] = system.strip() if system is not None else self.info.get('system', DEFAULT_SYSTEM_MESSAGE)
        self.save()

    @property
    def chat_id(self) -> str:
        """Unique identifier of the chat memory.

        Returns:
            str: Unique identifier of the chat memory.
        """
        return self._chat_id

    @property
    def title(self) -> str:
        """Chat title.

        Returns:
            str: Chat title.
        """
        title = self.info.get('title')
        if title is None:
            self.info['title'] = 'New Chat'
            self.save()
        return title
    
    @property
    def chat_dir(self) -> str:
        """Directory of the chat.

        Returns:
            str: Directory of the chat.
        """
        chat_dir = os.path.join(chat_memory_home(), self.chat_id)
        if not os.path.exists(chat_dir):
            os.makedirs(chat_dir)
        return chat_dir
    
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
            self._info = dict(title='New Chat', last_update=current_time())
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
    def history_dict(self) -> List[Dict[str, Any]]:
        """Entire history as dictionaries.

        Returns:
            List[Dict[str, Any]]: Entire history as dictionaries.
        """
        import gc
        history: list[dict[str, Any]] = list(map(lambda x: x.metadata, self._data.values()))
        if len(history) == 0:
            return []
        count = max(list(map(lambda x: x['order'], history))) + 1
        history = list(map(lambda x: list(filter(lambda y: y['order'] == x, history))[0], range(count)))
        history.sort(key=lambda x: x['order'], reverse=False)
        def reformat_record(record: dict[str, Any]) -> List[Dict[str, Any]]:
            user = dict(role='user', content=record['user'])
            assistant = dict(role='assistant', content=record['assistant'])
            for k, v in record.items():
                if k not in ['user', 'assistant', 'order', 'role']:
                    assistant[k] = v
            return [user, assistant]
        history = list(map(reformat_record, history))
        gc.collect()
        return sum(history, [])


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
        if ((from_exist) & (self.chat_id in list_chat_ids())):
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

    def update_title(self, title: str) -> None:
        """Update the title of the memory.

        Args:
            title (str): New chat memory title.
        """
        title = title.strip()
        if title == '':
            raise ValueError('Chat title cannot be an empty string.')
        self.info['title'] = title
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

