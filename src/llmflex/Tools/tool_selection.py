from .base_tool import BaseTool
from ..VectorDBs.faiss_vectordb import FaissVectorDatabase
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..Prompts.prompt_template import PromptTemplate, DEFAULT_SYSTEM_MESSAGE
from ..Models.Factory.llm_factory import LlmFactory
from ..Models.Cores.base_core import BaseLLM
from typing import Type, Optional, List, Union, Optional

TOOL_SELECTION_PROMPT = """Given a user request, your task is to select the most appropriate tool from the list provided below that can best address the request. If you believe the user's request can be adequately addressed without the use of any specific tool, indicate "no_tool_required". Please provide your response in JSON format, including a "toolName" for the selected tool name.

Below are a few examples of how to structure your response:

{examples}Please follow the above examples to structure your response for the user request.

These are the available tools and their respective description:
{tools}

Your response should be in JSON format, based on the tools available for above. If no tool is required, indicate "no_tool_required".

Here is the latest user request:

User Request: "{user_input}"
"""
TOOL_RPOMPT_START = """```json
{
    ""toolName": """ + '"'

TOOL_INFO = """Tool Name: "{tool_name}"
Description: "{description}"

"""

EXAMPLE_SHOT = """User Request: "[User request here]"
Response:
```json
{
    "toolName": "{tool_name}"
}
```

"""

class ToolSelector:
    """Class to select the appropriate tool given the user input."""

    def __init__(self, tools: List[Type[BaseTool]], model: LlmFactory, embeddings: Type[BaseEmbeddingsToolkit], score_threshold: float = 0.7) -> None:
        if len(tools) == 0:
            raise ValueError(f'At least one tool must be provided.')
        self._tools = tools
        self._vectordb = FaissVectorDatabase.from_documents(embeddings=embeddings, docs=[])
        self._llm = model(temperature=0, max_new_tokens=128, stop=model.prompt_template.stop + ['```'])
        self._score_threshold = score_threshold
        self._last_score = 0
        self._last_tool = ''
        example_count = 0
        examples = ''
        tool_info = ''
        for i, tool in enumerate(self.tools):
            index = [tool.name, tool.description] + tool.key_phrases
            data = dict(name=tool.name, description=tool.description)
            self.vectordb.add_texts(index, metadata=data)
            if example_count < 2:
                example_count += 1
                examples += EXAMPLE_SHOT.replace('{tool_name}', tool.name)
            tool_info += f'{i + 1}. ' + TOOL_INFO.replace('{tool_name}', tool.name).replace("{description}", tool.description)
        examples += EXAMPLE_SHOT.replace('{tool_name}', 'no_tool_required')
        self._examples = examples
        self._tool_info = tool_info


    @property
    def tools(self) -> List[BaseTool]:
        """List of tools.

        Returns:
            List[BaseTool]: List of tools.
        """
        return self._tools

    @property
    def vectordb(self) -> Optional[FaissVectorDatabase]:
        """Vector database for that store tools information.

        Returns:
            Optional[FaissVectorDatabase]: Vector database for that store tools information.
        """
        return self._vectordb
    
    @property
    def llm(self) -> Optional[BaseLLM]:
        """LLM used for tool selection.

        Returns:
            Optional[BaseLLM]: LLM used for tool selection.
        """
        return self._llm

    @property
    def score_threshold(self) -> float:
        """Score threshold for retrieving tools from vector database.

        Returns:
            float: Score threshold.
        """
        return self._score_threshold
    
    @property
    def last_score(self) -> float:
        """The last similarity score when using get_tool.

        Returns:
            float: The last similarity score when using get_tool.
        """
        return self._last_score
    
    @property
    def last_tool(self) -> str:
        """The last tool picked regardless of the score.

        Returns:
            str: The last tool picked regardless of the score.
        """
        return self._last_tool
    
    def set_score_threshold(self, score_threshold: float) -> None:
        """Set a new score threshold.

        Args:
            score_threshold (float): New score threshold.
        """
        self._score_threshold = score_threshold

    def get_tool(self, user_input: str, 
                 history: Union[List[str], List[List[str]]] = [], prompt_template: Optional[PromptTemplate] = None, system: str = DEFAULT_SYSTEM_MESSAGE) -> Optional[BaseTool]:
        """Select the most appropriate tool for a given user input.

        Args:
            user_input (str): User input string.
            history (Union[List[str], List[List[str]]], optional): Current conversation history. Defaults to [].
            prompt_template (Optional[PromptTemplate], optional): Prompt template to format the prompt. If None is provide, the llm prompt template will be used. Defaults to None.
            system (str, optional): System message of the current conversation. Defaults to DEFAULT_SYSTEM_MESSAGE.

        Returns:
            Optional[BaseTool]: Selected tool or None if no tool is found or required.
        """
        import json
        prompt_template = self.llm.core.prompt_template if prompt_template is None else prompt_template
        input = TOOL_SELECTION_PROMPT.replace("{examples}", self._examples).replace("{tools}", self._tool_info).replace("{user_input}", user_input)
        prompt = prompt_template.create_prompt(user=input, system=system, history=history) + TOOL_RPOMPT_START
        output = '{"tool": "' + self.llm.invoke(prompt)
        try:
            tool = json.loads(output)['tool']
        except:
            print('Tool selection unsuccessful.')
            tool = 'no_tool_required'
        if tool == 'no_tool_required':
            return None
        elif tool not in list(map(lambda x: x.name, self.tools)):
            print('Tool selection not in tool list.')
            return None
        else:
            etool = self.vectordb.search(user_input, top_k=1, index_only=False)[0]
            self._last_score = etool['score']
            self._last_tool = tool
            if ((etool['score'] >= self.score_threshold) & (etool['metadata']['name'] == tool)):
                tool = list(filter(lambda x: x.name == etool['metadata']['name'], self.tools))[0]
                return tool
            return None

            

