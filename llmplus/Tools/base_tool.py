from ..Prompts.prompt_template import PromptTemplate
from ..Models.Cores.base_core import BaseLLM
from typing import List, Iterator, Optional, Union

class BaseTool:
    """This is a base class for callables for LLMs.
    """
    def __init__(self, name: str = 'base_tool', description: str = 'This is a tool from the base tool class. It does not do anything.') -> None:
        """Initialising the tool.
        """
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        """Name of the tool.

        Returns:
            str: Name of the tool.
        """
        return self._name
    
    @property
    def description(self) -> str:
        """Description of the tool.

        Returns:
            str: Description of the tool.
        """
        import re
        newlines = re.compile(r'[\s\r\n\t]+')
        return newlines.sub(' ', self._description)
    
    def run(self, tool_input: str, llm: BaseLLM, stream: bool = False, 
            history: Optional[List[List[str]]] = None, prompt_template: Optional[PromptTemplate] = None, **kwargs) -> Union[str, Iterator[str]]:
        """Run the tool and return the output as a string.

        Args:
            tool_input (str): String input for to run the tool.
            llm (BaseLLM): LLM to generate the output.
            stream (bool, optional): Whether to stream the output, if True, a generator of the output will be returned. Defaults to False.
            history (Optional[List[List[str]]], optional): Snippet of chat history to help running the tool if required. Defaults to None.
            prompt_template (Optional[PromptTemplate], optional): prompt_template to format the chat history. Defaults to None.

        Returns:
            Union[str, Iterator[str]]: Output of the tool.
        """
        if ((history is not None) & (prompt_template is None)):
            raise ValueError('Prompt template need to be provided to process chat history.')
        return 'Base tool output.'