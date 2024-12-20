from __future__ import annotations
from .base_memory import BaseMemory
from typing import Optional, List, Dict, Any
import os

class ChatMemory(BaseMemory):
    """Basic chat memory class."""
    def __init__(self, title: Optional[str] = None, memory_dir: Optional[str] = None) -> None:
        """Initializes the ChateMemory class.

        Args:
            title (Optional[str], optional): The title of the memory. Defaults to None.
            memory_dir (Optional[str], optional): The directory where the memory will be stored. Defaults to None.
        """
        super().__init__(title, memory_dir)

    @classmethod
    def from_exist(cls, memory_dir: str) -> ChatMemory:
        """Loads a memory instance from an existing directory.

        Args:
            memory_dir (str): The directory containing the memory data.

        Returns:
            ChatMemory: An instance of the ChatMemory class.
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
        
        if os.path.exists(os.path.join(memory_dir, 'history.json')):
            with open(os.path.join(memory_dir, 'info.json'), 'r') as f:
                history = json.load(f)
        else:
            history = []
        memory = cls(title, memory_dir)
        memory._history = history
        return memory
        

    def _update_history(self, messages: List[Dict[str, Any]]) -> None:
        """Updates the conversation history with the given messages.

        Args:
            messages (List[Dict[str, Any]]): A list of dictionaries, each representing a conversation turn.
        """
        import json
        self.history.extend(messages)
        with open(os.path.join(self.memory_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

    def _remove_last_message(self) -> None:
        """Remove the latest message.
        """
        import json
        if len(self.history) > 0:
            self._history = self.history[:-1]
        with open(os.path.join(self.memory_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)