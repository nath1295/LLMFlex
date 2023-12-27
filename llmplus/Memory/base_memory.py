import os
from ..utils import get_config
from ..Prompts.prompt_template import PromptTemplate, presets
from ..Models.Cores.base_core import BaseLLM
from typing import List, Dict, Any, Union


def chat_memory_home() -> str:
    """Return the default directory for saving chat memories.

    Returns:
        str: The default directory for saving chat memories.
    """
    history_dir = os.path.join(get_config()['llmplus_home'], 'chat_memories')
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
    def __init__(self, title: str, from_exist: bool = True) -> None:
        """Initialising the memory class.

        Args:
            title (str): Title of the chat.
            from_exist (bool, optional): Initialising the chat memory from existing files if the title exists. Defaults to True.
        """
        title = title.strip(' \n\r\t')
        if title == '':
            raise ValueError('Chat title cannot be an empty string.')
        self._title = title
        self._init_memory(from_exist=from_exist)

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
    def history(self) -> List[List[str]]:
        """Entire chat history.

        Returns:
            List[List[str]]: Entire chat history.
        """
        history = list(map(lambda x: [x['metadata']['user'], x['metadata']['assistant'], x['metadata']['order']], self._data))
        if len(history) == 0:
            return []
        import pandas as pd
        history = pd.DataFrame(history, columns=['user', 'assistant', 'order'])
        history = history.drop_duplicates(subset=['order'], keep='first')
        return history.sort_values(by='order', ascending=True)[['user', 'assistant']].values.tolist()
    
    @property
    def interaction_count(self) -> int:
        return len(self.history)

    def _init_memory(self, from_exist: bool = True) -> None:
        """Method to initialise the components in the memory.

        Args:
            from_exist (bool, optional): Whether to initialise from existing files. Defaults to True.
        """
        if ((from_exist) & (self.title in list_titles())):
            from ..utils import read_json
            self._data = read_json(os.path.join(self.chat_dir, 'data.json'))

        else:
            self._data = list()
            self.save()


    def save(self) -> None:
        """Save the current state of the memory.
        """
        from ..utils import save_json, current_time
        self.info
        self._info['last_update'] = current_time()
        save_json(self._info, os.path.join(self.chat_dir, 'info.json'))
        save_json(self._data, os.path.join(self.chat_dir, 'data.json'))

    def save_interaction(self, user_input: str, assistant_output: str) -> None:
        """Saving an interaction to the memory.

        Args:
            user_input (str): User input.
            assistant_output (str): Chatbot output.
        """
        user_input = user_input.strip(' \n\r\t')
        assistant_output = assistant_output.strip(' \n\r\t')
        self._data.append(dict(
            index=f'{user_input}\n\n{assistant_output}',
            metadata=dict(user=user_input, assistant=assistant_output, order=self.interaction_count)
        ))
        self.save()

    def remove_last_interaction(self) -> None:
        """Remove the latest interaction.
        """
        if len(self._data) != 0:
            self._data = self._data[:-1]
            self.save()

    def clear(self) -> None:
        """Empty the whole chat history.
        """
        self._init_memory(from_exist=False)

    def get_recent_memory(self, k: int = 3) -> List[List[str]]:
        """Get the last k interactions as a list.

        Args:
            k (int, optional): Maximum number of latest interactions. Defaults to 3.

        Returns:
            List[List[str]]: List of interactions.
        """
        history = self.history
        results = history if len(history) <= k else history[-k:]
        return results

    def get_token_memory(self, llm: BaseLLM, token_limit: int = 400) -> List[List[str]]:
        """Get the latest conversation limited by number of tokens.

        Args:
            llm (BaseLLM): LLM to count tokens.
            token_limit (int, optional): Maximum number of tokens allowed. Defaults to 400.

        Returns:
            List[List[str]]: The formatted output of the recent conversation.
        """
        if len(self.history) == 0:
            return []
        tk_count = 0
        history = list(reversed(self.history))
        results = list()
        for m in history:
            msg_count = sum(list(map(lambda x: llm.get_num_tokens(x), m)))
            if (msg_count + tk_count) <= token_limit:
                results = [m] + results
                tk_count += msg_count
            else:
                break
        if len(results) == 0:
            results = [history[0]]
        return results





