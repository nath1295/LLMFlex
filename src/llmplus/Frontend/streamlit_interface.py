from ..Models.Factory.llm_factory import LlmFactory
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from ..Embeddings.huggingface_embeddings import HuggingfaceEmbeddingsToolkit
from ..Embeddings.api_embeddings import APIEmbeddingsToolkit
from ..Tools.base_tool import BaseTool
from ..utils import PACKAGE_DISPLAY_NAME
import streamlit as st
from typing import Dict, Any, Union, List, Tuple, Type, Optional, Literal, Iterator


class InterfaceState:

    def __init__(self, model: LlmFactory, embeddings: Type[BaseEmbeddingsToolkit], tools: List[Type[BaseTool]] = []) -> None:
        """Initialise the backend of the Streamlit interface.

        Args:
            model (LlmFactory): LLM factory.
            embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit.
            tools (List[Type[BaseTool]], optional): List of tools. Defaults to [].
        """
        from ..Memory.assistant_long_term_memory import AssistantLongTermChatMemory
        from ..Tools.tool_selection import ToolSelector
        from ..TextSplitters.sentence_token_text_splitter import SentenceTokenTextSplitter
        from ..Prompts.prompt_template import DEFAULT_SYSTEM_MESSAGE
        self.model = model
        self.embeddings = embeddings
        self.text_splitter = SentenceTokenTextSplitter(count_token_fn=model().get_num_tokens, chunk_size=200, chunk_overlap=40)
        self.memory = AssistantLongTermChatMemory(title='Untitled 0', embeddings=self.embeddings, text_splitter=self.text_splitter, from_exist=False)
        self.system = DEFAULT_SYSTEM_MESSAGE
        self.template = self.model.prompt_template
        self.llm = self.model(stop=self.template.stop + ['#####'])
        self.short_limit = 600
        self.long_limit = 500
        self.score_threshold = 0.5
        self.tool_selector = ToolSelector(tools, model=self.model, embeddings=self.embeddings) if len(tools) > 0 else None

    @property
    def titles(self) -> List[str]:
        """All existing chat titles.

        Returns:
            List[str]: All existing chat titles.
        """
        from ..Memory.base_memory import list_titles
        return list_titles()
    
    @property
    def current_title(self) -> str:
        """Current memory chat title.

        Returns:
            str: Current memory chat title.
        """
        return self.memory.title
      
    @property
    def history(self) -> List[List[str]]:
        """Current conversation history.

        Returns:
            List[List[str]]: Current conversation history.
        """
        return self.memory.history
       
    @property
    def presets(self) -> List[str]:
        """List of prompt templates presets.

        Returns:
            List[str]: List of prompt templates presets.
        """
        from ..Prompts.prompt_template import presets
        return list(presets.keys())

class StreamlitInterface:

    def __init__(self, model_kwargs: Dict[str, Any], 
                 embeddings_kwargs: Dict[str, Any],
                 tool_kwargs: List[Dict[str, Any]] = [],
                 auth: Optional[Tuple[str, str]] = None, 
                 debug: bool = False) -> None:
        if not hasattr(st.session_state, 'backend'):
            model = LlmFactory(**model_kwargs)
            embeddings = embeddings_loader(embeddings_kwargs)
            tools = list(map(lambda x: tool_loader(x, embeddings, model), tool_kwargs))
            st.session_state.backend = InterfaceState(model, embeddings, tools=tools)
        self.debug = debug
        self._auth = auth
        if auth is None:
            st.session_state.islogin = True
        self.sidebar_ratio = [5, 1]

    @property
    def backend(self) -> InterfaceState:
        return st.session_state.backend
        
    @property
    def islogin(self) -> bool:
        if not hasattr(st.session_state, 'islogin'):
            st.session_state.islogin = False
        return st.session_state.islogin
    
    @property
    def mobile(self) -> bool:
        if not hasattr(st.session_state, 'mobile'):
            st.session_state.mobile = False
        return st.session_state.mobile

    @property
    def login_wrong(self) -> bool:
        if not hasattr(st.session_state, 'login_wrong'):
            st.session_state.login_wrong = False
        return st.session_state.login_wrong

    @property
    def conversation_delete(self) -> bool:
        if not hasattr(st.session_state, 'conversation_delete'):
            st.session_state.conversation_delete = False
        return st.session_state.conversation_delete

    @property
    def generating(self) -> bool:
        """Whether chatbot is generating."""
        if not hasattr(st.session_state, 'generating'):
            st.session_state.generating = False
        return st.session_state.generating
    
    @property
    def experimental(self) -> bool:
        if not hasattr(st.session_state, 'allow_ai_start'):
            st.session_state.allow_ai_start = False
        return st.session_state.allow_ai_start

    @property
    def history_dict(self) -> List[Dict[str, Any]]:
        if not hasattr(st.session_state, 'history'):
            self.refresh_history()
        elif st.session_state.history['current_title'] != self.backend.current_title:
            self.refresh_history()
        return st.session_state.history['history_dict']
    
    @property
    def generation_config(self) -> Dict[str, str]:
        if not hasattr(st.session_state, 'generation_config'):
            st.session_state.generation_config = dict(gen_type='none')
        return st.session_state.generation_config
    
    @property
    def ai_start_text(self) -> str:
        if not hasattr(st.session_state, 'ai_start_text'):
            st.session_state.ai_start_text = ''
        return st.session_state.ai_start_text

    @property
    def generation_time_info(self) -> str:
        template = '  \nGeneration time taken: '
        if not hasattr(st.session_state, 'generation_time_info'):
            st.session_state.generation_time_info = '--'
        if type(st.session_state.generation_time_info) == str:
            return template + st.session_state.generation_time_info
        else:
            return template + f'{round(st.session_state.generation_time_info, 2)}' + 's'

    @property
    def tool_states(self) -> Dict[str, bool]:
        if not hasattr(st.session_state, 'tool_states'):
            st.session_state.tool_states = dict()
            for tool in self.backend.tool_selector.tools:
                st.session_state.tool_states[tool.name] = True
        return st.session_state.tool_states

    def get_tool(self, user_input: str) -> Optional[Type[BaseTool]]:
        if self.backend.tool_selector is None:
            return None
        if sum(list(self.tool_states.values())) == 0:
            return None
        self.backend.tool_selector.set_score_threshold(self.tool_threshold)
        history = self.backend.memory.get_token_memory(llm=self.backend.llm, token_limit=self.backend.short_limit)
        system = self.backend.system
        tool = self.backend.tool_selector.get_tool(user_input=user_input, history=history, system=system)
        print(f'Tool: {self.backend.tool_selector.last_tool}\nScore: {self.backend.tool_selector.last_score}')
        if tool is not None:
            if not self.tool_states[tool.name]:
                print(tool.name + ' disabled.')
                tool = None
        return tool
    
    def run_tool(self, tool: BaseTool, user_input: str) -> Iterator[Union[str, Tuple[str, str], Iterator[str]]]:
        recent_history = self.backend.memory.get_token_memory(llm=self.backend.llm, token_limit=self.backend.short_limit)
        return tool.run_with_chat(tool_input=user_input, llm=self.backend.llm, stream=True, history=recent_history)

    def login_with_cred(self, user: str, password: str) -> None:
        if ((user == self._auth[0]) & (password == self._auth[1])):
            print(user, password)
            st.session_state.islogin = True
        else:
            st.session_state.login_wrong = True
        st.rerun()

    def toggle_generating(self) -> None:
        st.session_state.generating = not self.generating
        # print(f'Generating = {self.generating}')

    def toggle_mobile(self) -> None:
        st.session_state.mobile = not self.mobile

    def toggle_tool(self, tool_name: str) -> None:
        if not self.generating:
            self.tool_states[tool_name] = not self.tool_states[tool_name]

    def toggle_conversation_delete(self) -> None:
        st.session_state.conversation_delete = not self.conversation_delete

    def set_mobile(self) -> None:
        if not self.generating:
            st.session_state.mobile = st.session_state.mobile_toggle

    def current_prompt_index(self) -> int:
        from ..Prompts.prompt_template import presets
        template = self.backend.model.prompt_template.template_name
        for i, t in enumerate(self.backend.presets):
            if template == t:
                return i
        return 0
    
    def set_time_info(self, time: Union[str, float] = '--') -> None:
        st.session_state.generation_time_info = time     

    def add_chat(self, title: str) -> None:
        if not self.generating:
            title = title.strip(' \r\n\t')
            if title == '':
                pass
            else:
                from ..Memory.assistant_long_term_memory import AssistantLongTermChatMemory
                from ..Prompts.prompt_template import DEFAULT_SYSTEM_MESSAGE
                self.backend.memory = AssistantLongTermChatMemory(title, embeddings=self.backend.embeddings, text_splitter=self.backend.text_splitter, from_exist=True)
                self.backend.system = self.backend.memory.vectordb._info.get('system', DEFAULT_SYSTEM_MESSAGE)
                self.backend.memory.vectordb._info['system'] = self.backend.system
                self.backend.memory.save()
                st.rerun()

    def switch_chat(self, title: str) -> None:
        if not self.generating:
            from ..Memory.assistant_long_term_memory import AssistantLongTermChatMemory
            from ..Prompts.prompt_template import DEFAULT_SYSTEM_MESSAGE
            st.session_state.generating = True
            self.set_time_info()
            if title == self.backend.current_title:
                pass
            else:
                print(f'Switch to: {title}')
                from_exist = title != 'Untitled 0'
                self.backend.memory = AssistantLongTermChatMemory(title=title, embeddings=self.backend.embeddings, text_splitter=self.backend.text_splitter, from_exist=from_exist)
                self.backend.system = self.backend.memory.info.get('system', DEFAULT_SYSTEM_MESSAGE)
                self.backend.memory.save()
            st.session_state.generating = False

    def delete_chat(self, title: str) -> None:
        if not self.generating:
            switch = title == self.backend.current_title
            from shutil import rmtree
            if switch:
                rmtree(self.backend.memory.chat_dir)
                self.switch_chat('Untitled 0')
            else:
                from ..Memory.assistant_long_term_memory import AssistantLongTermChatMemory
                mem = AssistantLongTermChatMemory(title, embeddings=self.backend.embeddings, text_splitter=self.backend.text_splitter, from_exist=False)
                rmtree(mem.chat_dir)
        
    def toggle_exeperimental(self) -> None:
        if not self.generating:
            st.session_state.allow_ai_start = not st.session_state.allow_ai_start

    def set_exeperimental(self) -> None:
        if not self.generating:
            st.session_state.allow_ai_start = st.session_state.use_ai_start

    def set_prompt_template(self) -> None:
        if not self.generating:
            from ..Prompts.prompt_template import PromptTemplate
            preset = st.session_state.prompt_format
            self.backend.template = PromptTemplate.from_preset(preset)

    def set_system_message(self, system: str) -> None:
        if not self.generating:
            self.backend.system = system.strip(' \n\r\t')
            self.backend.memory.vectordb._info['system'] = self.backend.system
            self.backend.memory.save()

    def set_memory_settings(self, long: int, short: int, score: float) -> None:
        if not self.generating:
            self.backend.short_limit = short
            self.backend.long_limit = long
            self.backend.score_threshold = score

    def set_llm_config(self, temperature: float, max_new_tokens: int, repetition_penalty: float, top_p: float, top_k: int) -> None:
        if not self.generating:
            self.backend.llm = self.backend.model(temperature=temperature, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k)

    def get_history(self, mem) -> List[Dict[str, Any]]:
        history = list(map(lambda x: dict(
            user=x['metadata']['user'], 
            assistant=x['metadata']['assistant'], 
            order=x['metadata']['order'],
            tool_details=x['metadata'].get('tool_details', None),
            footnote=x['metadata'].get('footnote', None),
            tool_name=x['metadata'].get('tool_name', None)
        ), mem._data))
        if len(history) == 0:
            return []
        count = max(list(map(lambda x: x['order'], history))) + 1
        history = list(map(lambda x: list(filter(lambda y: y['order'] == x, history))[0], range(count)))
        history.sort(key=lambda x: x['order'], reverse=False)
        return history

    def refresh_history(self) -> None:
        st.session_state.history = dict(
            current_title=self.backend.current_title,
            history_dict=self.get_history(self.backend.memory)
        )

    def save_interaction(self, user: str, assistant: str, **kwargs) -> None:
        self.backend.memory.save_interaction(user_input=user, assistant_output=assistant, **kwargs)
        self.refresh_history()

    def input_template(self, user: Optional[str], ai_start: str) -> Optional[Dict[str, Any]]:
        if user is None:
            return None
        user = user.strip(' \n\r\t')
        if user == '':
            return None
        input_dict = dict(
            user = user.strip(' \n\r\t'),
            assistant = ai_start,
            order = self.backend.memory.interaction_count,
            tool_details = None,
            footnote = None,
            tool_name = None
        )
        return input_dict

    def create_generation_config(self, gen_type: Literal['new', 'retry', 'continue', 'none'], user_input: str, ai_start: Optional[str] = None) -> None:
        gen_obj = dict(
            gen_type = gen_type,
            user_input = user_input.strip(' \r\n\t'),
            ai_start = self.ai_start if ai_start is None else ai_start
        )
        st.session_state.generation_config = gen_obj

    def get_generation_iterator(self) -> Iterator[str]:
        from ..Memory.assistant_long_term_memory import create_long_assistant_memory_prompt
        config = self.generation_config
        prompt = create_long_assistant_memory_prompt(
            user=config['user_input'],
            prompt_template=self.backend.template,
            llm=self.backend.llm,
            memory=self.backend.memory,
            system=self.backend.system,
            short_token_limit=self.backend.short_limit,
            long_token_limit=self.backend.long_limit,
            score_threshold=self.backend.score_threshold
        ) + config['ai_start']
        print(f'Number of input tokens: {self.backend.llm.get_num_tokens(prompt)}')
        def generator():
            yield config['ai_start']
            for i in self.backend.llm.stream(prompt):
                yield i
        return generator()

    def retry_response(self, cont: bool = False, ai_start: str = '') -> None:
        if not self.generating:
            if len(self.history_dict) == 0:
                return None
            print(ai_start)
            st.session_state.ai_start_text = ai_start
            history = self.get_history(self.backend.memory)
            user = history[-1]['user']
            self.create_generation_config('continue' if cont else 'retry', user, history[-1]['assistant'] if cont else ai_start)
            input_dict = self.input_template(user=user, ai_start=history[-1]['assistant'] if cont else ai_start)
            self.backend.memory.remove_last_interaction()
            self.refresh_history()
            self.history_dict.append(input_dict)
            self.toggle_generating()

    def remove_last(self) -> None:
        if not self.generating:
            self.backend.memory.remove_last_interaction()
            self.refresh_history()

    ##### Defining the interface
        
    def login(self) -> None:
        login_form = st.form(key='login')
        with login_form:
            user = st.text_input(label='Username:', placeholder='Your username...')
            password = st.text_input(label='Password', placeholder='Your password...', type='password')
            if st.form_submit_button(label='Login'):
                print('Triggered by btn.')
                self.login_with_cred(user=user, password=password)
            if self.login_wrong:
                st.warning('Incorrect credentials. Please try again.')

    def sidebar(self) -> None:
        """Sidebar of the webapp.
        """
        app_summary = ['Powered by:', 
                       f'* LLM: {self.backend.model.model_id}', 
                       f'* Embedding model: {self.backend.embeddings.name}', 
                       '', 
                       'Current conversation:', 
                       f'* {self.backend.current_title}',
                       '',
                       'Current prompt format:',
                       f'* {self.backend.template.template_name}']
        st.header('LLM Plus', help='  \n'.join(app_summary), divider="grey")
        st.subheader(':left_speech_bubble: Conversations')
        self.new_chat_form()
        with st.expander(label='Previous conversations'):
            self.conversations()
        self.settings()
        if self.debug:
            st.subheader('Debug')
            self.test_buttons()

    def new_chat_form(self) -> None:
        with st.form(key='new_chat_form', border=False, clear_on_submit=True):
            cols = st.columns(self.sidebar_ratio)
            with cols[0]:
                self.new_title = st.text_input(label='new_title', max_chars=40, placeholder='New conversation title here...', label_visibility='collapsed', disabled=self.generating)
            with cols[1]:
                if st.form_submit_button(label=':heavy_plus_sign:', disabled=self.generating):
                    self.add_chat(title=self.new_title)

    def conversations(self) -> None:
        """List of conversations."""
        for title in self.backend.titles:
            if self.conversation_delete:
                cols = st.columns(self.sidebar_ratio)
                with cols[0]:
                    btn_type = 'primary' if self.backend.current_title == title else 'secondary'
                    st.button(label=title.title(), key=f'{title}_select', disabled=self.generating, 
                            use_container_width=True, type=btn_type, on_click=self.switch_chat, kwargs=dict(title=title))
                with cols[1]:
                    st.button(label=':heavy_minus_sign:', key=f'{title}_delete', disabled=self.generating, on_click=self.delete_chat, kwargs=dict(title=title))
            else:
                btn_type = 'primary' if self.backend.current_title == title else 'secondary'
                st.button(label=title.title(), key=f'{title}_select', disabled=self.generating, 
                        use_container_width=True, type=btn_type, on_click=self.switch_chat, kwargs=dict(title=title))
        cols = st.columns(self.sidebar_ratio)
        with cols[0]:
            st.toggle(label=':wastebasket:', key=f'conv_delete', disabled=self.generating, on_change=self.toggle_conversation_delete, help='Select conversations to remove.')

    def test_buttons(self) -> None:
        cols = st.columns([1, 1])
        with cols[0]:
            st.button('Test print', key='test', on_click=lambda: print(self.ai_start))

        with cols[1]:
            st.button('gen_toggle', key='toggle_gen', on_click=lambda: self.toggle_generating())

    def settings(self) -> None:
        """Settings of the webapp."""
        st.subheader(':gear: Settings')
        st.toggle(label='Experimental', 
                    value=self.experimental,
                    disabled=self.generating,
                    on_change=self.set_exeperimental,
                    key='use_ai_start', 
                    help='More features such as retrying, adding response starting message etc.')
        st.toggle(label='Mobile', 
                    value=self.mobile,
                    disabled=self.generating,
                    on_change=self.set_mobile,
                    key='mobile_toggle', 
                    help='Better layout for mobile device.')
        with st.expander(label='Prompt format settings'):
            self.prompt_template_settings()
        with st.expander(label='System message settings'):
            self.system_prompt_settings()
        with st.expander(label='Memory settings'):
            self.memory_settings()
        with st.expander(label='Model settings'):
            self.llm_settings()
        self.tool_settings()

    def prompt_template_settings(self) -> None:
        """Prompt template settings."""
        from ..Prompts.prompt_template import presets
        format = st.selectbox(label='prompt_formats', 
                     label_visibility='collapsed',
                     key='prompt_format',
                     options=list(presets.keys()), 
                     disabled=self.generating, 
                     index=self.current_prompt_index(),
                     on_change=self.set_prompt_template)
        
    def system_prompt_settings(self) -> None:
        """System prompt settings."""
        self.system_text = st.text_area(label='System message', height=250, key='system_msg',label_visibility='collapsed', value=self.backend.system, disabled=self.generating)
        st.markdown(f'System message token count: {self.backend.llm.get_num_tokens(self.backend.system)}')
        st.button(label=':floppy_disk:', key='system_save', disabled=self.generating, use_container_width=True, on_click=self.set_system_message, kwargs=dict(system=self.system_text))

    def memory_settings(self) -> None:
        """Memory token limit settings."""
        self.short_limit_slidder = st.slider('Short term memory token limit', min_value=0, max_value=6000, step=1, value=self.backend.short_limit, disabled=self.generating)
        self.long_limit_slidder = st.slider('Long term memory token limit', min_value=0, max_value=6000, step=1, value=self.backend.long_limit, disabled=self.generating)
        self.score_threshold_slidder = st.slider('Score threshold for long term memory', min_value=0.0, max_value=1.0, step=0.01, value=self.backend.score_threshold, disabled=self.generating)
        summary = [
            'Current settings:',
            f'Short term memory token limit: {self.backend.short_limit}',
            f'Long term memory token limit: {self.backend.long_limit}',
            f'Score threshold: {self.backend.score_threshold}'

        ]
        st.markdown('  \n'.join(summary))
        st.button(label=':floppy_disk:', key='memory_token_save', disabled=self.generating, 
                  use_container_width=True,
                  on_click=self.set_memory_settings,
                  kwargs=dict(short=self.short_limit_slidder,
                              long=self.long_limit_slidder,
                              score=self.score_threshold_slidder))

    def llm_settings(self) -> None:
        """LLM generation settings."""
        self.temperature_slidder = st.slider('Temparature', min_value=0.0, max_value=2.0, step=0.01, value=self.backend.llm.generation_config['temperature'], disabled=self.generating)
        self.max_new_token_slidder = st.slider('Maximum number of new tokens', min_value=0, max_value=4096, step=1, value=self.backend.llm.generation_config['max_new_tokens'], disabled=self.generating)
        self.repetition_slidder = st.slider('Repetition penalty', min_value=1.0, max_value=2.0, step=0.01, value=self.backend.llm.generation_config['repetition_penalty'], disabled=self.generating)
        self.topp_slidder = st.slider('Top P', min_value=0.0, max_value=1.0, step=0.01, value=self.backend.llm.generation_config['top_p'], disabled=self.generating)
        self.topk_slidder = st.slider('Top K', min_value=0, max_value=30000, step=1, value=self.backend.llm.generation_config['top_k'], disabled=self.generating)
        summary = [
            'Current settings:',
            f"Temperature: {self.backend.llm.generation_config['temperature']}",
            f"Max new tokens: {self.backend.llm.generation_config['max_new_tokens']}",
            f"Repetition penalty: {self.backend.llm.generation_config['repetition_penalty']}",
            f"Top P: {self.backend.llm.generation_config['top_p']}",
            f"Top K: {self.backend.llm.generation_config['top_k']}",
        ]
        st.markdown('  \n'.join(summary))
        st.button(label=':floppy_disk:', 
                  key='llm_config_save', 
                  disabled=self.generating, 
                  use_container_width=True,
                  on_click=self.set_llm_config,
                  kwargs=dict(temperature=self.temperature_slidder,
                              max_new_tokens=self.max_new_token_slidder,
                              repetition_penalty=self.repetition_slidder,
                              top_p=self.topp_slidder,
                              top_k=self.topk_slidder))
        
    def tool_settings(self) -> None:
        if self.backend.tool_selector is not None:
            with st.expander('Tools settings'):
                for i in self.backend.tool_selector.tools:
                    st.toggle(label=f'{i.pretty_name}', value=self.tool_states[i.name], on_change=lambda: self.toggle_tool(i.name),
                              disabled=self.generating)
                self.tool_threshold = st.slider(label='Tool trigger threshold', min_value=0.0, max_value=1.0, step=0.005, 
                                                value=self.backend.tool_selector.score_threshold, disabled=self.generating)

    def chatbot(self) -> None:
        self.conversation_history()
        
        self.ai_start_textbox()
        self.user_input_box()

    def conversation_history(self) -> None:
        history = self.history_dict
        last = len(history) - 1
        gen_type = self.generation_config['gen_type']
        for i, ex in enumerate(history):
            with st.chat_message(name='user'):
                st.markdown(ex['user'], help=f'Number of tokens: {self.backend.llm.get_num_tokens(ex["user"])}')
            with st.chat_message(name='assistant'):
                self.assistant_response(ex, i, last)
            if ((gen_type != 'none') & (i == last)):
                st.session_state.generation_config = dict(gen_type='none')
                st.session_state.retry_or_continue = False
                self.toggle_generating()
                st.rerun()

    def assistant_response(self, ex: Dict[str, Any], i: int, last: int) -> None:
        if ex['tool_details'] is not None:
            with st.status(label=f":hammer_and_pick: __{ex['tool_name']}__", state='complete'):
                st.code(ex['tool_details'][1], language='plaintext')
        md = ex['assistant']
        if ex['footnote'] is not None:
            md += '\n\n---\n' + ex['footnote']
        if ((not self.generating) | (i!=last)):
            help_info = f'Number of tokens: {self.backend.llm.get_num_tokens(ex["assistant"])}'
            if i==last:
                help_info += self.generation_time_info
            st.markdown(md, help=help_info)
        else:
            # st.button(label=':black_square_for_stop:', help='Stop response generation')
            with st.spinner('Thinking....'):
                from time import perf_counter
                start = perf_counter()
                tool = self.get_tool(user_input=ex['user'])
                if tool is None:
                    placeholder = st.empty()
                    self.output = ''
                    for i in self.get_generation_iterator():
                        self.output += i
                        placeholder.markdown(self.output.strip(' \r\n\t'))
                    self.history_dict[-1]['assistant'] = self.output.strip(' \r\n\t')
                    self.save_interaction(user=self.history_dict[-1]['user'], assistant=self.history_dict[-1]['assistant'])
                    end = perf_counter() - start
                    self.set_time_info(end)
                    print(f'Number of output tokens: {self.backend.llm.get_num_tokens(self.output)}')
                else:
                    tool_name = tool.pretty_name
                    tool_details = None
                    footnote = None
                    toolholder = st.empty()
                    placeholder = st.empty()
                    md_text = ''
                    for chunk in self.run_tool(tool=tool, user_input=ex['user']):
                        if isinstance(chunk, tuple):
                            tool_details = chunk
                            with toolholder.status(label=f":hammer_and_pick: Running __{tool_name}__...", state='running'):
                                st.text(tool_details[1])
                        elif isinstance(chunk, str):
                            footnote = chunk
                            md_text += '\n\n---\n' + chunk
                            placeholder.markdown(md_text)
                        else:
                            with toolholder.status(label=f":hammer_and_pick: __{tool_name}__", state='complete'):
                                st.text(tool_details[1])
                            output = ''
                            for i in chunk:
                                output += i
                                md_text = output.strip(' \n\r\t')
                                placeholder.markdown(md_text)
                            end = perf_counter() - start
                            self.set_time_info(end)
                            print(f'Number of output tokens: {self.backend.llm.get_num_tokens(output)}')
                    self.save_interaction(user=ex['user'], assistant=output.strip(' \r\n\t'), tool_details=tool_details, footnote=footnote, tool_name=tool_name)
                    

                            
                
            
    def experimental_buttons(self) -> None:
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.button(':arrows_counterclockwise:', use_container_width=True, help='Re-generate response', disabled=self.generating, 
                      on_click=self.retry_response, kwargs=dict(cont=False, ai_start=self.ai_start))
        with cols[1]:
            st.button(':fast_forward:', use_container_width=True, help='Continue generating response', disabled=self.generating,
                      on_click=self.retry_response, kwargs=dict(cont=True, ai_start=self.ai_start))
        with cols[2]:
            st.button(':wastebasket:', use_container_width=True, help='Remove the latest question and response', disabled=self.generating, on_click=self.remove_last)

    def ai_start_textbox(self) -> None:
        if ((self.experimental) & (not self.generating)):
            with st.container(border=False):
                self.ai_start = st.text_area(label='AI start', placeholder='Start of the chatbot response here...', value=self.ai_start_text, height=1, label_visibility='collapsed')
                if self.mobile:
                    with st.expander(':gear: Extra options'):
                        self.experimental_buttons()
                else:
                    self.experimental_buttons()
            
        else:
            self.ai_start = ''

    def user_input_box(self) -> None:
        self.user_input = st.chat_input(placeholder='Your message...', disabled=self.generating)
        if self.user_input:
            if self.user_input.strip(' \r\n\t') != '':
                input_dict = self.input_template(user=self.user_input, ai_start=self.ai_start)
                self.history_dict.append(input_dict)
                self.create_generation_config('new', self.user_input)
                self.toggle_generating()
                st.rerun()
            
    def launch(self) -> None:
        if self.islogin:
            if not self.mobile:
                st.set_page_config(layout='wide')
            else:
                st.set_page_config(layout='centered')
            with st.sidebar:
                self.sidebar()
            self.chatbot()
        else:
            self.login()

def embeddings_loader(embeddings_kwargs: Dict[str, Any]) -> BaseEmbeddingsToolkit:
    """Load the embeddings given the kwargs.

    Args:
        embeddings_kwargs (Dict[str, Any]): Kwargs to initialise the embeddings toolkit.

    Returns:
        BaseEmbeddingsToolkit: The embeddings toolkit.
    """
    mapper = {"HuggingfaceEmbeddingsToolkit": HuggingfaceEmbeddingsToolkit, 'APIEmbeddingsToolkit': APIEmbeddingsToolkit}
    model_key = embeddings_kwargs.pop('embeddings_class', "HuggingfaceEmbeddingsToolkit")
    return mapper.get(model_key)(**embeddings_kwargs)

def tool_loader(tool_kwargs: Dict[str, Any], embeddings: Type[BaseEmbeddingsToolkit], model: LlmFactory) -> BaseTool:
    """Load the embeddings given the kwargs.

    Args:
        tool_kwargs (Dict[str, Any]): Kwargs to initialise the tool.
        embeddings (Type[BaseEmbeddingsToolkit]): Embeddings toolkit for the tool if needed.
        model (LlmFactory): LlmFactory for the tool if needed.

    Returns:
        BaseTool: The tool.
    """
    from ..Tools.web_search_tool import WebSearchTool
    mapper = {'WebSearchTool': WebSearchTool}
    tool_class = mapper.get(tool_kwargs.pop('tool_class', None))
    if tool_class == None:
        raise ValueError('"tool_class" must be given.')
    if tool_kwargs.pop('model', False):
        tool_kwargs['model'] = model
    if tool_kwargs.pop('embeddings', False):
        tool_kwargs['embeddings'] = embeddings
    return tool_class(**tool_kwargs)

def create_streamlit_script(model_kwargs: Dict[str, Any], 
                 embeddings_kwargs: Dict[str, Any],
                 tool_kwargs: List[Dict[str, Any]] = [],
                 auth: Optional[Tuple[str, str]] = None, 
                 debug: bool = False) -> str:
    """Create the script to run the streamlit interface.

    Args:
        model_kwargs (Dict[str, Any]): Kwargs to initialise the LLM factory.
        embeddings_kwargs (Dict[str, Any]): Kwargs to initialise the embeddings toolkit.
        tool_kwargs (List[Dict[str, Any]], optional): List of kwargs to initialise the tools. Defaults to [].
        auth (Optional[Tuple[str, str]], optional): Tuple of username and password. Defaults to None.
        debug (bool, optional): Whether to display the debug buttons. Defaults to False.

    Returns:
        str: The streamlit script as a string.
    """
    from ..utils import PACKAGE_NAME
    script = [f'from {PACKAGE_NAME}.Frontend.streamlit_interface import StreamlitInterface', '']
    script.append(f'model = {str(model_kwargs)}')
    script.append(f'embeddings = {str(embeddings_kwargs)}')
    script.append(f'tools = {str(tool_kwargs)}')
    script.append(f'auth = {auth}')
    script.append(f'debug = {debug}')
    script.append('')
    script.append('app = StreamlitInterface(model, embeddings, tools, auth, debug)\napp.launch()')
    return '\n'.join(script)

def run_streamlit_interface(model_kwargs: Dict[str, Any], 
                 embeddings_kwargs: Dict[str, Any],
                 tool_kwargs: List[Dict[str, Any]] = [],
                 auth: Optional[Tuple[str, str]] = None, 
                 debug: bool = False,
                 app_name: str = PACKAGE_DISPLAY_NAME) -> None:
    """Run the streamlit interface.

    Args:
        model_kwargs (Dict[str, Any]): Kwargs to initialise the LLM factory.
        embeddings_kwargs (Dict[str, Any]): Kwargs to initialise the embeddings toolkit.
        tool_kwargs (List[Dict[str, Any]], optional): List of kwargs to initialise the tools. Defaults to [].
        auth (Optional[Tuple[str, str]], optional): Tuple of username and password. Defaults to None.
        debug (bool, optional): Whether to display the debug buttons. Defaults to False.
        app_name (str, optional): name of the streamlit script created. Defaults to PACKAGE_DISPLAY_NAME.
    """
    import subprocess
    import os
    from ..utils import get_config
    script_dir = os.path.join(get_config()['package_home'], '.streamlit_scripts',f'{app_name}.py')
    os.makedirs(os.path.dirname(script_dir), exist_ok=True)
    with open(script_dir, 'w') as f:
        f.write(create_streamlit_script(model_kwargs, embeddings_kwargs, tool_kwargs, auth, debug))
    os.system('streamlit run '+ script_dir)