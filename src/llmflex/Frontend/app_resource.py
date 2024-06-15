from ..Models.Factory.llm_factory import LlmFactory
from ..Models.Cores.base_core import BaseLLM
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..TextSplitters.base_text_splitter import BaseTextSplitter
from ..Rankers.base_ranker import BaseRanker
from ..Tools.tool_utils import BaseTool, ToolSelector, normalise_tool_name
from ..Prompts.prompt_template import presets, PromptTemplate
from ..Memory.long_short_memory import LongShortTermChatMemory
from ..Memory.base_memory import list_chat_ids, get_new_chat_id, get_dir_from_id
from ..KnowledgeBase.knowledge_base import KnowledgeBase
from typing import List, Type, Dict, Any, Union, Optional

class AppBackend:
    """Resources for the App to share.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the backend resourses.
        Args:
            config (Dict[str, Any]): Configuration of all the resources.
        """
        self._config = config
        self._init_llm_factory()
        self._init_ranker()
        self._init_text_splitter()
        self._init_embeddings()
        self._init_tools()
        self._knowledge_base = None
        
    @property
    def config(self) -> Dict[str, Any]:
        """Configuration of all the resources.ry_

        Returns:
            Dict[str, Any]: Configuration of all the resources.
        """
        return self._config
    
    @property
    def factory(self) -> LlmFactory:
        """LLM factory.

        Returns:
            LlmFactory: LLM factory.
        """
        return self._factory
    
    @property
    def llm(self) -> BaseLLM:
        """LLM.

        Returns:
            BaseLLM: LLM.
        """
        if not hasattr(self, '_llm'):
            self._llm = self.factory()
        return self._llm
    
    @property
    def ranker(self) -> BaseRanker:
        """Reranker.

        Returns:
            BaseRanker: Reranker.
        """
        return self._ranker
    
    @property
    def text_splitter(self) -> BaseTextSplitter:
        """Text splitter.

        Returns:
            BaseTextSplitter: Text splitter.
        """
        return self._text_splitter
    
    @property
    def embeddings(self) -> BaseEmbeddingsToolkit:
        """Embeddings toolkit.

        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit.
        """
        return self._embeddings
    
    @property
    def tool_selector(self) -> ToolSelector:
        """Tool selector.

        Returns:
            ToolSelector: Tool selector.
        """
        return self._tool_selector
    
    @property
    def prompt_template(self) -> PromptTemplate:
        """Prompt template.

        Returns:
            PromptTemplate: Prompt template.
        """
        if not hasattr(self, '_prompt_template'):
            self._prompt_template = self.factory.prompt_template
        return self._prompt_template

    @property
    def memory(self) -> LongShortTermChatMemory:
        """Current chat memory.

        Returns:
            LongShortTermChatMemory: Current chat memory.
        """
        if not hasattr(self, '_memory'):
            self.create_memory()
        return self._memory
    
    @property
    def generation_config(self) -> Dict[str, float]:
        """Text generation config.

        Returns:
            Dict[str, float]: Text generation config.
        """
        if not hasattr(self, '_generation_config'):
            self._generation_config = dict(
            temperature = 0.8,
            max_new_tokens = 1024,
            top_p  = 0.95,
            top_k = 40,
            repetition_penalty = 1.1
        )
        return self._generation_config

    @property
    def memory_config(self) -> Dict[str, float]:
        """Memory extraction config.

        Returns:
            Dict[str, float]: Memory extraction config.
        """
        if not hasattr(self, '_memory_config'):
            self._memory_config = dict(
                recent_token_limit = 600, 
                relevant_token_limit= 500,
                relevance_score_threshold = 0.8, 
                similarity_score_threshold = 0.5
            )
        return self._memory_config
    
    @property
    def knowledge_base_config(self) -> Dict[str, float]:
        """Knowledge base search config.

        Returns:
            Dict[str, float]: Knowledge base search config.
        """
        if not hasattr(self, '_knowledge_base_config'):
            self._knowledge_base_config = dict(
                kb_token_limit = 500,
                kb_score_threshold = 0.0
            )
        return self._knowledge_base_config

    @property
    def tool_status(self) -> Dict[str, bool]:
        """Whether each tool is on or off.

        Returns:
            Dict[str, bool]: Whether each tool is on or off.
        """
        from copy import deepcopy
        import gc
        status = deepcopy(self.tool_selector._enabled)
        status.pop('direct_response')
        gc.collect()
        return status

    @property
    def has_tools(self) -> bool:
        """Whether any tools exist in the tool selector.

        Returns:
            bool: Whether any tools exist in the tool selector.
        """
        if not hasattr(self, '_has_tools'):
            self._has_tools = len(self.tool_status) > 0
        return self._has_tools
    
    @property
    def knowledge_base_map_dir(self) -> str:
        """Directory of knowledge base mapping json file.

        Returns:
            str: Directory of knowledge base mapping json file.
        """
        if not hasattr(self, '_knowledge_base_map_dir'):
            import os
            from ..utils import get_config
            phome = get_config()['package_home']
            self._knowledge_base_map_dir = os.path.join(phome, '.streamlit_scripts', 'knowledge_base_map.json')
            os.makedirs(os.path.dirname(self._knowledge_base_map_dir), exist_ok=True)
        return self._knowledge_base_map_dir
    
    @property
    def knowledge_base_map(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """Dictionary of knowledge bases.

        Returns:
            Dict[str, Dict[str, Union[str, List[str]]]]: Dictionary of knowledge bases.
        """
        from ..KnowledgeBase.knowledge_base import list_knowledge_base
        from ..utils import read_json
        import os
        kbs = list_knowledge_base()
        kb_map = dict()
        if os.path.exists(self.knowledge_base_map_dir):
            kb_map_temp = read_json(self.knowledge_base_map_dir)
        else:
            kb_map_temp = dict()
        for kb_id in kbs:
            kb_map[kb_id] = dict(title=kb_map_temp.get(kb_id, dict(title=kb_id))['title'], chat_ids=kb_map_temp.get(kb_id, dict(chat_ids=[]))['chat_ids'])
        return kb_map
            
    @property
    def knowledge_base(self) -> Optional[KnowledgeBase]:
        """Knowledge base.

        Returns:
            Optional[KnowledgeBase]: Knowledge base.
        """
        if not hasattr(self, '_knowledge_base'):
            self._knowledge_base = None
        return self._knowledge_base

    def _init_llm_factory(self) -> None:
        """Initialising the llm factory.
        """
        self._factory = LlmFactory(**self.config['model'])

    def _init_ranker(self) -> None:
        """Initialise the reranker.
        """
        from .. import Rankers
        from copy import deepcopy
        config = deepcopy(self.config.get('ranker', dict(class_name='FlashrankRanker')))
        class_name = config.pop('class_name', 'FlashrankRanker')
        rclass = Rankers.__dict__[class_name]
        self._ranker = rclass(**config)

    def _init_text_splitter(self) -> None:
        """Initialise the text splitter.
        """
        from .. import TextSplitters
        from copy import deepcopy
        config = deepcopy(self.config.get('text_splitter', dict(class_name='SentenceTokenTextSplitter', count_token_fn='default')))
        class_name = config.pop('class_name', 'SentenceTokenTextSplitter')
        if config.get('count_token_fn') == 'default':
            config['count_token_fn'] = self.llm.get_num_tokens
        rclass = TextSplitters.__dict__[class_name]
        self._text_splitter = rclass(**config)      

    def _init_embeddings(self) -> None:
        """Initialising the embeddings toolkit.
        """
        from .. import Embeddings
        from copy import deepcopy
        config = deepcopy(self.config.get('embeddings', dict(class_name='HuggingfaceEmbeddingsToolkit', model_id='')))
        class_name = config.pop('class_name', 'HuggingfaceEmbeddingsToolkit')
        if ((class_name == 'HuggingfaceEmbeddingsToolkit') & ('model_id' not in config.keys())):
            config['model_id'] = ''
        eclass = Embeddings.__dict__[class_name]
        self._embeddings= eclass(**config)

    def _init_tools(self) -> None:
        """Initialising tool selector.
        """
        from ..Tools import tool_classes
        from copy import deepcopy
        from inspect import isclass
        config = deepcopy(self.config.get('tools', []))
        tool_classes =  list(map(lambda x: tool_classes.__dict__.get(x['class_name']), config))
        tool_names = []
        tool_instances = []
        for i, cf in enumerate(config):
            tclass = tool_classes[i]
            tool_name = normalise_tool_name(cf.pop('class_name'))
            if isclass(tclass):
                if issubclass(tclass, BaseTool):
                    if cf.get('llm') == 'default':
                        cf['llm'] = self.llm
                    if cf.get('embeddings') == 'default':
                        cf['embeddings'] = self.embeddings
                    if cf.get('ranker') == 'default':
                        cf['ranker'] = self.ranker
                    if cf.get('text_splitter') == 'default':
                        cf['text_splitter'] = self.text_splitter
                    if cf.get('count_token_fn') == 'default':
                        cf['count_token_fn'] = self.llm.get_num_tokens
                    tool = tclass(**cf)
                    if tool.name not in tool_names:
                        tool_instances.append(tool)
                        tool_names.append(tool.name)
            elif tool_name not in tool_names:
                tool_instances.append(tclass)
                tool_names.append(tool_name)
        self._tool_selector = ToolSelector(tools=tool_instances)
        self.tool_selector.turn_off_tools(self.tool_selector.enabled_tools)

    def switch_memory(self, chat_id: str) -> None:
        """Switch to the memory given the chat ID.

        Args:
            chat_id (str): Chat ID.
        """
        if chat_id in list_chat_ids():
            import gc
            self._memory = LongShortTermChatMemory(chat_id=chat_id, 
                    embeddings=self.embeddings, 
                    llm=self.llm, 
                    ranker=self.ranker, 
                    text_splitter=self.text_splitter,
                    from_exist=True)
            del self._knowledge_base
            self._knowledge_base = None
            gc.collect()
            for k, v in self.knowledge_base_map.items():
                if chat_id in v['chat_ids']:
                    self._knowledge_base = KnowledgeBase(kb_id=k, embeddings=self.embeddings, llm=self.llm, ranker=self.ranker, text_splitter=self.text_splitter) 
                    break
            
    def create_memory(self) -> None:
        """Create a new chat memory.
        """
        import gc
        self._memory = LongShortTermChatMemory(chat_id=get_new_chat_id(), 
                embeddings=self.embeddings, 
                llm=self.llm, 
                ranker=self.ranker, 
                text_splitter=self.text_splitter)
        del self._knowledge_base
        self._knowledge_base = None
        gc.collect()

    def drop_memory(self, chat_id: str) -> None:
        """Delete the chat memory give the chat ID.

        Args:
            chat_id (str): Chat ID.
        """
        import shutil
        if chat_id in list_chat_ids():
            if chat_id == self.memory.chat_id:
                self.create_memory()
            kb_map = self.knowledge_base_map
            kb_items = list(kb_map.items())
            for k, v in kb_items:
                if chat_id in v['chat_ids']:
                    kb_map[k]['chat_ids'].remove(chat_id)
                    from ..utils import save_json
                    save_json(kb_map, self.knowledge_base_map_dir)
            chat_dir = get_dir_from_id(chat_id=chat_id)
            shutil.rmtree(chat_dir)

    def set_generation_config(self, temperature: Optional[float] = None, max_new_tokens: Optional[int] = None, 
                top_p: Optional[float] = None, top_k: Optional[int] = None, repetition_penalty: Optional[float] = None) -> None:
        """Update the LLM generation config. If None is given to any arguments, the argument will not change.

        Args:
            temperature (Optional[float], optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to None.
            max_new_tokens (Optional[int], optional): Maximum number of tokens to generate by the llm. Defaults to None.
            top_p (Optional[float], optional): While sampling the next token, only consider the tokens above this p value. Defaults to None.
            top_k (Optional[int], optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to None.
            repetition_penalty (Optional[float], optional): The value to penalise the model for generating repetitive text. Defaults to None.
        """
        args = [temperature, max_new_tokens, top_p, top_k, repetition_penalty]
        arg_names = ['temperature', 'max_new_tokens', 'top_p', 'top_k', 'repetition_penalty']
        for i, arg in enumerate(args):
            if arg is not None:
                self._generation_config[arg_names[i]] = arg

    def set_memory_config(self, recent_token_limit: Optional[int] = None, relevant_token_limit: Optional[int] = None, 
                relevance_score_threshold: Optional[float] = None, similarity_score_threshold: Optional[float] = None) -> None:
        """Update the memory config. If None is given to any arguments, the argument will not change.

        Args:
        recent_token_limit (Optional[int], optional): Token limit for the most recent conversation history. Defaults to None.
        relevant_token_limit (Optional[int], optional): Token limit for the relevant contents from older conversation history. Defaults to None.
        relevance_score_threshold (Optional[float], optional): Score threshold for the reranker for relevant conversation history content extraction. Defaults to None.
        similarity_score_threshold (Optional[float], optional): Score threshold for the vector database search for relevant conversation history content extraction. Defaults to None.
        """
        args = [recent_token_limit, relevant_token_limit, relevance_score_threshold, similarity_score_threshold]
        arg_names = ['recent_token_limit', 'relevant_token_limit', 'relevance_score_threshold', 'similarity_score_threshold']
        for i, arg in enumerate(args):
            if arg is not None:
                self._memory_config[arg_names[i]] = arg

    def set_knowledge_base_config(self, kb_token_limit: Optional[int] = None, kb_score_threshold: Optional[float] = None) -> None:
        """Update the knowledge base config. If None is given to any arguments, the argument will not change.

        Args:
            kb_token_limit (Optional[int], optional): Token limit for the search. Defaults to None.
            kb_score_threshold (Optional[float], optional): Score threshold for the reranker for knowledge base search. Defaults to None.
        """
        args = [kb_token_limit, kb_score_threshold]
        arg_names = ['kb_token_limit', 'kb_score_threshold']
        for i, arg in enumerate(args):
            if arg is not None:
                self._knowledge_base_config[arg_names[i]] = arg

    def set_system_message(self, system: str) -> None:
        """Update the system message of the current conversation.

        Args:
            system (str): Update the system message of the current conversation.
        """
        self.memory.update_system_message(system=system)

    def set_prompt_template(self, preset: str) -> None:
        """Updating prompt template.

        Args:
            preset (str): Preset name of the prompt template.
        """
        if preset in presets.keys():
            self._prompt_template = PromptTemplate.from_preset(style=preset)

    def toggle_tool(self, tool_name: str) -> None:
        """Toggle the on/off status of the given tool.

        Args:
            tool_name (str): Tool to toggle.
        """
        if tool_name in self.tool_status.keys():
            self.tool_selector._enabled[tool_name] = not self.tool_selector._enabled[tool_name]

    def create_knowledge_base(self, title: str, files: List[str]) -> None:
        """Create and return a knowledge base for a chat given the files.

        Args:
            title (str): Title of the knowledge base.
            files (List[str]): Files to create the knowledge base.
        """
        from ..KnowledgeBase.knowledge_base import get_new_kb_id, load_file, list_knowledge_base
        from ..utils import save_json
        self.detach_knowledge_base()
        kb = KnowledgeBase(kb_id=get_new_kb_id(), embeddings=self.embeddings, llm=self.llm, ranker=self.ranker, text_splitter=self.text_splitter)
        kb_map = self.knowledge_base_map
        kb_map[kb.kb_id] = dict(title=title.strip(), chat_ids=[self.memory.chat_id])
        save_json(kb_map, file_dir=self.knowledge_base_map_dir)
        for file in files:
            try:
                kb.add_documents(docs=load_file(file_dir=file))
            except:
                print(f'File "{file}" cannot be loaded.')
        self._knowledge_base = kb

    def select_knowledge_base(self, kb_id: str) -> None:
        """Attach current chat memory to an existing knowledge base.

        Args:
            kb_id (str): Knowledge base ID to attach.
        """
        from ..KnowledgeBase.knowledge_base import list_knowledge_base
        from ..utils import save_json
        import gc
        if kb_id in list_knowledge_base():
            if self.knowledge_base is None:
                kb_map = self.knowledge_base_map
                kb_map[kb_id]['chat_ids'].append(self.memory.chat_id)
                save_json(kb_map, self.knowledge_base_map_dir)
                del self._knowledge_base
                self._knowledge_base = KnowledgeBase(kb_id=kb_id, embeddings=self.embeddings, llm=self.llm, ranker=self.ranker, text_splitter=self.text_splitter)
                gc.collect()
            elif kb_id != self.knowledge_base.kb_id:
                kb_map = self.knowledge_base_map
                kb_map[self.knowledge_base.kb_id]['chat_ids'].remove(self.memory.chat_id)
                kb_map[kb_id]['chat_ids'].append(self.memory.chat_id)
                del self._knowledge_base
                self._knowledge_base = KnowledgeBase(kb_id=kb_id, embeddings=self.embeddings, llm=self.llm, ranker=self.ranker, text_splitter=self.text_splitter)
                gc.collect()

    def detach_knowledge_base(self) -> None:
        """Detach current knowledge base.
        """
        if self.knowledge_base is not None:
            from ..utils import save_json
            import gc
            kb_map = self.knowledge_base_map
            kb_map[self.knowledge_base.kb_id]['chat_ids'].remove(self.memory.chat_id)
            save_json(kb_map, self.knowledge_base_map_dir)
            self._knowledge_base = None
            gc.collect()

    def remove_knowledge_base(self, kb_id: str) -> None:
        """Remove the knowledge base.

        Args:
            kb_id (str): Knowledge base ID to remove.
        """
        from ..KnowledgeBase.knowledge_base import list_knowledge_base, knowledge_base_dir
        if self.knowledge_base is not None:
            if self.knowledge_base.kb_id == kb_id:
                import gc
                self._knowledge_base = None
                gc.collect()
        if kb_id in list_knowledge_base():
            from shutil import rmtree
            import os
            rmtree(os.path.join(knowledge_base_dir(), kb_id))


    
