from __future__ import annotations
from abc import ABC, abstractmethod
import os
from datetime import datetime as dt
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from ..Tokenizer.base_tokenizer import BaseTokenizer

class BaseMemory(ABC):
    """Base chat memory class."""
    def __init__(self, title: Optional[str] = None, memory_dir: Optional[str] = None) -> None:
        """Initializes the BaseMemory class.

        Args:
            title (Optional[str], optional): The title of the memory. Defaults to None.
            memory_dir (Optional[str], optional): The directory where the memory will be stored. Defaults to None.
        """
        self._title = title
        self._memory_dir = memory_dir
        self._save_info()

    @property
    def title(self) -> Optional[str]:
        """Returns the title of the memory.

        Returns:
            Optional[str]: The title of the memory.
        """
        return self._title
    
    @property
    def memory_dir(self) -> Optional[str]:
        """Returns the directory where the memory will be stored.

        Returns:
            Optional[str]: The directory where the memory will be stored.
        """
        if self._memory_dir:
            os.makedirs(self._memory_dir, exist_ok=True)
        return self._memory_dir
    
    @property
    def info(self) -> Dict[str, Union[str, Optional[str], float]]:
        """Returns information about the memory.

        Returns:
            Dict[str, Union[str, Optional[str], float]]: A dictionary containing information about the memory.
        """
        if not hasattr(self, '_info'):
            self._info = dict(
                memory_class=self.__class__.__name__,
                title=self.title,
                last_update=dt.now().timestamp()
            )
        return self._info
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """Returns the conversation history.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a conversation turn.
        """
        if not hasattr(self, '_history'):
            self._history = []
        return self._history
    
    def set_title(self, title: Optional[str]) -> None:
        """Sets the title of the memory.

        Args:
            title (Optional[str]): The new title for the memory.
        """
        self._title = title
        self.info['title'] = title
        self._save_info()

    def _save_info(self) -> None:
        """Saves the information about the memory to a file."""
        self.info['last_update'] = dt.now().timestamp()
        if self.memory_dir:
            import json
            with open(os.path.join(self.memory_dir, 'info.json'), 'w') as f:
                json.dump(self.info, f, indent=4)

    @classmethod
    @abstractmethod
    def from_exist(cls, memory_dir: str) -> BaseMemory:
        """Loads a memory instance from an existing directory.

        Args:
            memory_dir (str): The directory containing the memory data.

        Returns:
            BaseMemory: An instance of the BaseMemory subclass.
        """
        pass

    @abstractmethod
    def _update_history(self, messages: List[Dict[str, Any]]) -> None:
        """Updates the conversation history with the given messages.

        Args:
            messages (List[Dict[str, Any]]): A list of dictionaries, each representing a conversation turn.
        """
        pass

    @abstractmethod
    def _remove_last_message(self) -> None:
        """Remove the latest message.
        """
        pass

    def remove_last_message(self) -> None:
        """Remove the latest message.
        """
        self._remove_last_message()
        self._save_info()

    def save_messages(self, messages: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """Saves the given messages to the memory.

        Args:
            messages (Union[Dict[str, Any], List[Dict[str, Any]]]): A dictionary or list of dictionaries representing the messages to be saved.
        """
        from ..Prompt.chat_template import ChatMessage
        # Validating messages
        save_messages = [messages] if isinstance(messages, dict) else messages
        for msg in save_messages:
            ChatMessage(**msg)
        self._update_history(save_messages)
        self._save_info()

    def get_last_n_messages(self, last_n: int = 3) -> List[Dict[str, Any]]:
        """Returns the last n messages from the conversation history.

        Args:
            last_n (int, optional): The number of messages to retrieve. Defaults to 3.

        Raises:
            ValueError: If last_n is less than 1.

        Returns:
            List[Dict[str, Any]]: A list of the last n messages.
        """
        if last_n < 1:
            raise ValueError('"last_n" must be larger than or equal to 1.')
        history = self.history[-last_n:]
        if (len(history) > 0) and (history[0]['role'] == 'tool'):
            history = self.history[-(last_n + 1):]
        return history
    
    def get_messages_by_token_limit(self, tokenizer: "BaseTokenizer", token_limit: int) -> List[Dict[str, Any]]:
        """Returns messages from the conversation history that do not exceed the given token limit.

        Args:
            tokenizer (BaseTokenizer): The tokenizer to use for token counting.
            token_limit (int): The maximum number of tokens allowed in the returned messages.

        Returns:
            List[Dict[str, Any]]: A list of messages that do not exceed the token limit.
        """
        import json
        from copy import deepcopy
        token_count = 0
        reversed_history = self.history[::-1]
        if len(reversed_history) == 0:
            return []

        messages = []
        for msg in reversed_history:
            content = msg.get('content', None)
            content = '' if content is None else content
            tool_calls = msg.get('tool_calls', None)
            tool_calls = [] if tool_calls is None else tool_calls
            content_token_count = 0 if content == '' else tokenizer.get_num_tokens(content, add_special_tokens=False)
            tool_calls_token_count = 0
            for tool_call in tool_calls:
                arguments = tool_call['function']['arguments']
                try:
                    arg_dict = json.loads(arguments)
                except:
                    arg_dict = arguments
                tc_dict = deepcopy(tool_call)
                tc_dict['function']['arguments'] = arg_dict
                tc_str = json.dumps(tc_dict)
                tool_calls_token_count += tokenizer.get_num_tokens(tc_str, add_special_tokens=False)
            if (content_token_count + tool_calls_token_count + token_count) < token_limit:
                messages = [msg] + messages
                token_count += content_token_count + tool_calls_token_count
            else:
                break
        
        # Make sure at least one message is returned if the history has any messages.
        if len(messages) == 0:
            messages = [reversed_history[0]]
        return messages


