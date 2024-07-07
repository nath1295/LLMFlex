import streamlit as st
from streamlit_extras.bottom_container import bottom
from streamlit_extras.row import row
from ..Frontend.app_resource import AppBackend
from ..Memory.base_memory import list_chat_ids, get_title_from_id
from ..Memory.memory_utils import create_prompt_with_history
from ..Prompts.prompt_template import presets
from ..Tools.tool_utils import gen_string
from ..utils import PACKAGE_DISPLAY_NAME
from typing import Dict, Any, Literal, Optional, Tuple, List
import yaml

DEFAULT_CONFIG = dict(
    model = dict(
        model_id = 'NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF',
        model_type = 'gguf',
        model_file = 'Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf',
        context_length = 8192
    ),
    embeddings = dict(
        class_name = 'HuggingfaceEmbeddingsToolkit',
        model_id = 'thenlper/gte-small'
    ),
    ranker = dict(
        class_name = 'FlashrankRanker'
    ),
    text_splitter = dict(
        class_name = 'SentenceTokenTextSplitter',
        count_token_fn = 'default'
    ),
    tools = [
        dict(
            class_name = 'BrowserTool',
            llm = 'default',
            embeddings = 'default',
            ranker = 'default'
        ),
        dict(
            class_name = 'math_tool'
        )
    ],
    credentials = None
)

def create_streamlit_script() -> str:
    """Create the main script to run the app.

    Returns:
        str: Script directory.
    """
    script = ['from llmflex.Frontend.streamlit_interface import AppInterface']
    script.append('if __name__ == "__main__":')
    script.append('\tapp = AppInterface()\n\tapp.run()')
    script = '\n'.join(script)
    import os
    from ..utils import get_config
    script_dir = os.path.join(get_config()['package_home'], '.streamlit_scripts', 'webapp.py')
    os.makedirs(os.path.dirname(script_dir), exist_ok=True)
    with open(script_dir, 'w') as f:
        f.write(script)
    return script_dir

@st.cache_resource
def get_backend() -> AppBackend:
    """Get the backend object for the webapp.

    Returns:
        AppBackend: Backend object for the webapp.
    """
    from ..utils import get_config
    import os
    with open(os.path.join(get_config()['package_home'], '.streamlit_scripts', 'chatbot_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    return AppBackend(config=config)

class AppInterface:

    def __init__(self) -> None:
        from ..utils import get_config
        import os
        st.set_page_config(layout='wide')
        self.log_dir = os.path.join(get_config()['package_home'], '.streamlit_scripts', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def credentials(self) -> Optional[Tuple[str, str]]:
        """Login credentials if provided.

        Returns:
            Optional[Tuple[str, str]]: Login credentials if provided.
        """
        return self.backend.config.get('credentials')

    @property
    def is_login(self) -> bool:
        """Whether it is logged in or not.

        Returns:
            bool: Whether it is logged in or not.
        """
        if not hasattr(st.session_state, 'is_login'):
            st.session_state.is_login = False if self.credentials is not None else True
        return st.session_state.is_login

    @property
    def backend(self) -> AppBackend:
        """Backend resources.

        Returns:
            AppBackend: Backend resources.
        """
        return get_backend()
    
    @property
    def generating(self) -> bool:
        """Whether text generation in progress.

        Returns:
            bool: Whether text generation in progress.
        """
        if not hasattr(st.session_state, 'generating'):
            st.session_state.generating = False
        return st.session_state.generating
    
    @property
    def chat_delete_button(self) -> bool:
        """Whether to show chat deletion buttons.

        Returns:
            bool: Whether to show chat deletion buttons.
        """
        if not hasattr(st.session_state, 'chat_delete_button'):
            st.session_state.chat_delete_button = False
        return st.session_state.chat_delete_button
    
    @property
    def kb_delete_button(self) -> bool:
        """Whether to show knowledge base deletion buttons.

        Returns:
            bool: Whether to show knowledge base deletion buttons.
        """
        if not hasattr(st.session_state, 'kb_delete_button'):
            st.session_state.kb_delete_button = False
        return st.session_state.kb_delete_button

    @property
    def kb_create_button(self) -> bool:
        """Whether to show knowledge base creation buttons.

        Returns:
            bool: Whether to show knowledge base creation buttons.
        """
        if not hasattr(st.session_state, 'kb_create_button'):
            st.session_state.kb_create_button = False
        return st.session_state.kb_create_button

    @property
    def mobile(self) -> bool:
        """Whether on mobile device.

        Returns:
            bool: Whether on mobile device.
        """
        if not hasattr(st.session_state, 'mobile'):
            st.session_state.mobile = False
        return st.session_state.mobile
    
    @property
    def enable_begin_text(self) -> bool:
        """Whether on mobile device.

        Returns:
            bool: Whether on mobile device.
        """
        if not hasattr(st.session_state, 'enable_begin_text'):
            st.session_state.enable_begin_text = False
        return st.session_state.enable_begin_text
    
    @property
    def begin_text_cache(self) -> str:
        """Begin text cache.

        Returns:
            str: Begin text cache.
        """
        if not hasattr(st.session_state, 'begin_text_cache'):
            st.session_state.begin_text_cache = ''
        return st.session_state.begin_text_cache

    @property
    def input_config(self) -> Optional[Dict[str, Any]]:
        """Configuration for text generation if available.

        Returns:
            Optional[Dict[str, Any]]: Configuration for text generation if available.
        """
        if hasattr(st.session_state, 'input_config'):
            return st.session_state.input_config

    # Login page
    def login(self) -> None:
        """Creating login page.
        """
        login_form = st.form(key='login')
        with login_form:
            user = st.text_input(label='Username:', placeholder='Your username...')
            password = st.text_input(label='Password', placeholder='Your password...', type='password')
            if st.form_submit_button(label='Login'):
                if ((user == self.credentials[0]) & (password == self.credentials[1])):
                    st.session_state.is_login = True
                    st.rerun()
                else:
                    st.warning('Incorrect credentials. Please try again.')

    # Sidebar frontend
    def chats(self) -> None:
        """Listing all conversations.
        """
        st.button(label=':heavy_plus_sign: Start a new conversation', key=f'new_chat', use_container_width=True, on_click=self.backend.create_memory, disabled=self.generating)

        active_chat_id = self.backend.memory.chat_id
        ids = list_chat_ids()

        def toggle_delete_chats():
            st.session_state.chat_delete_button = not st.session_state.chat_delete_button

        if self.chat_delete_button:
            convs = dict()
            for id in ids:
                btn_type = 'primary' if id == active_chat_id else 'secondary'
                real_title = get_title_from_id(id)
                title = real_title.replace('_', ' ').title()
                if len(title) > 20:
                    title = title[:20] + '...'
                convs[id] = row(spec=[0.8, 0.2])
                convs[id].button(label=title, key=id, type=btn_type, use_container_width=True, 
                                 on_click=self.backend.switch_memory, kwargs=dict(chat_id=id), disabled=self.generating, help=real_title)
                convs[id].button(label=':heavy_minus_sign:', key=f'del_{id}', use_container_width=True, 
                                 on_click=self.backend.drop_memory, kwargs=dict(chat_id=id), disabled=self.generating)
        else:
            for id in ids:
                btn_type = 'primary' if id == active_chat_id else 'secondary'
                real_title = get_title_from_id(id)
                title = real_title.replace('_', ' ').title()
                if len(title) > 25:
                    title = title[:25] + '...'
                st.button(label=title, key=id, type=btn_type, use_container_width=True, 
                          on_click=self.backend.switch_memory, kwargs=dict(chat_id=id), disabled=self.generating, help=real_title)
        st.toggle(label=':wastebasket:', key=f'conv_delete', disabled=self.generating, 
                value=self.chat_delete_button, on_change=toggle_delete_chats, help='Select conversations to remove.')

    def knowledge_base_config(self) -> None:
        """Creating knowledge base configurations.
        """
        if self.kb_delete_button:
            kbs = dict()
            for k, v in self.backend.knowledge_base_map.items():
                btn_type = 'secondary'
                if self.backend.knowledge_base is not None:
                    if self.backend.knowledge_base.kb_id == k:
                        btn_type = 'primary'
                title = v['title'].replace('_', ' ').title()
                if len(title) > 25:
                    title = title[:25] + '...'
                kbs[k] = row(spec=[0.8, 0.2])
                kbs[k].button(label=title, key=k, type=btn_type, use_container_width=True, 
                                 on_click=self.backend.select_knowledge_base, kwargs=dict(kb_id=k), disabled=self.generating, help=v['title'])
                kbs[k].button(label=':heavy_minus_sign:', key=f'del_{k}', use_container_width=True, 
                                 on_click=self.backend.remove_knowledge_base, kwargs=dict(kb_id=k), disabled=self.generating)
        else:
            for k, v in self.backend.knowledge_base_map.items():
                btn_type = 'secondary'
                if self.backend.knowledge_base is not None:
                    if self.backend.knowledge_base.kb_id == k:
                        btn_type = 'primary'
                title = v['title'].replace('_', ' ').title()
                if len(title) > 25:
                    title = title[:25] + '...'
                st.button(label=title, key=k, type=btn_type, use_container_width=True, 
                          on_click=self.backend.select_knowledge_base, kwargs=dict(kb_id=k), disabled=self.generating, help=v['title'])

        def toggle_delete_kb():
            st.session_state.kb_delete_button = not st.session_state.kb_delete_button
        st.toggle(label=':wastebasket:', key=f'kb_delete', disabled=self.generating, 
                value=self.kb_delete_button, on_change=toggle_delete_kb, help='Select knowledge base to remove.')
        self.knowledge_base_creation()
        
    def knowledge_base_creation(self) -> None:
        """Creating knowledge base.
        """
        def toggle_create_kb():
            st.session_state.kb_create_button = not st.session_state.kb_create_button
        st.toggle(label=':heavy_plus_sign:', key=f'kb_create', disabled=self.generating, 
                value=self.kb_create_button, on_change=toggle_create_kb, help='Creating a new knowledge base.')
        if self.kb_create_button:
            import os
            new_kb_title = st.text_input(label='Knowledge Base Title', value='', disabled=self.generating)
            files = st.file_uploader(label='Upload files', disabled=self.generating, label_visibility='collapsed', accept_multiple_files=True, type=['pdf', 'txt', 'md', 'docx', 'py'])
            temp_files_dir = os.path.join(os.path.dirname(self.backend.knowledge_base_map_dir), 'temp_files')
            os.makedirs(temp_files_dir, exist_ok=True)
            def create_kb_fn():
                if new_kb_title.strip() != '':
                    file_dirs = []
                    if files:
                        for file in files:
                            file_dir = os.path.join(temp_files_dir, file.name)
                            with open(file_dir, 'wb') as f:
                                f.write(file.getvalue())
                            file_dirs.append(file_dir)
                        self.backend.create_knowledge_base(title=new_kb_title, files=file_dirs)
                    from shutil import rmtree
                    rmtree(temp_files_dir)
            create_btn = st.button(label='__Create__', disabled=self.generating, on_click=create_kb_fn)


    def settings(self) -> None:
        """Creating the settings.
        """
        page_dict = {
            'Prompt Format Settings': self.prompt_format_settings, 
            'System Message Settings': self.system_message_setttings, 
            'Memory Settings': self.memory_settings,
            'Knowledge Base Settings': self.knowledge_base_settings,
            'Model Settings': self.model_settings
            }
        if self.backend.has_tools:
            page_dict['Tool Settings'] = self.tool_settings
        self.mobile_settings()
        self.begin_text_settings()
        setting_dropdown = st.selectbox(
            label = 'Setting dropdown',
            label_visibility='collapsed',
            options= list(page_dict.keys())
        )
        st.markdown(setting_dropdown + ':')
        page_dict[setting_dropdown]()

    def mobile_settings(self) -> None:
        """Create toggle button for mobile mode.
        """
        def toggle_mobile() -> None:
            st.session_state.mobile = not st.session_state.mobile
        st.toggle(label='Mobile', value=self.mobile, help='Toggle mobile mode.', on_change=toggle_mobile, disabled=self.generating) 

    def begin_text_settings(self) -> None:
        """Create toggle button for response starting text.
        """
        def toggle_begin_text() -> None:
            st.session_state.enable_begin_text = not st.session_state.enable_begin_text
        st.toggle(label='Response Edit', value=self.enable_begin_text, help='Toggle mobile mode.', on_change=toggle_begin_text, disabled=self.generating) 

    def prompt_format_settings(self) -> None:
        """Prompt format settings.
        """
        def set_prompt_template() -> None:
            if st.session_state.prompt_format in presets.keys():
                self.backend.set_prompt_template(st.session_state.prompt_format)
            else:
                self.backend._prompt_template = self.backend.factory.prompt_template
        options = list(presets.keys())
        if self.backend.prompt_template.template_name not in options:
            options.append(self.backend.prompt_template.template_name)
        option_index_dict = dict(zip(options, range(len(options))))
        current_index = option_index_dict.get(self.backend.prompt_template.template_name)
        prompt_format = st.selectbox(
            label='prompt_formats', 
            label_visibility='collapsed',
            key='prompt_format',
            options=options, 
            disabled=self.generating, 
            index=current_index,
            on_change=set_prompt_template
        )
        if not self.backend.prompt_template.allow_custom_role:
            st.warning('Current prompt format does not support function calling.')
        st.markdown('Format Example')
        prompt_example = [
            dict(role='system', content='This is system message.'),
            dict(role='user', content='Hi there!'),
            dict(role='assistant', content='Hello to you too!'),
            dict(role='user', content='Shall we talk?')
            ]
        st.text(self.backend.prompt_template.create_custom_prompt(prompt_example))
        
    def system_message_setttings(self) -> None:
        """Create settings for system message.
        """
        self.system_text = st.text_area(label='System message', height=250, key='system_msg',label_visibility='collapsed', value=self.backend.memory.system, disabled=self.generating)
        st.markdown(f'System message token count: {self.backend.llm.get_num_tokens(self.backend.memory.system)}')
        st.button(label=':floppy_disk:', key='system_save', disabled=self.generating, use_container_width=True, on_click=self.backend.set_system_message, kwargs=dict(system=self.system_text))

    def memory_settings(self) -> None:
        """Create settings for memory.
        """
        self.short_limit_slidder = st.slider('Short term memory token limit', min_value=0, max_value=10000, step=1, 
                value=self.backend.memory_config['recent_token_limit'], disabled=self.generating)
        self.long_limit_slidder = st.slider('Long term memory token limit', min_value=0, max_value=6000, step=1, 
                value=self.backend.memory_config['relevant_token_limit'], disabled=self.generating)
        self.rel_score_threshold_slidder = st.slider('Relevance score threshold for long term memory', min_value=0.0, max_value=1.0, step=0.01,
                value=self.backend.memory_config['relevance_score_threshold'], disabled=self.generating)
        self.sim_score_threshold_slidder = st.slider('Similarity score threshold for long term memory', min_value=0.0, max_value=1.0, step=0.01,
                value=self.backend.memory_config['similarity_score_threshold'], disabled=self.generating)
        summary = [
            'Current settings:',
            f"Short term memory token limit: {self.backend.memory_config['recent_token_limit']}",
            f"Long term memory token limit: {self.backend.memory_config['relevant_token_limit']}",
            f"Relevance score threshold: {self.backend.memory_config['relevance_score_threshold']}",
            f"Similarity score threshold: {self.backend.memory_config['similarity_score_threshold']}"

        ]
        st.markdown('  \n'.join(summary))
        st.button(label=':floppy_disk:', key='memory_token_save', disabled=self.generating, 
                  use_container_width=True,
                  on_click=self.backend.set_memory_config,
                  kwargs=dict(recent_token_limit=self.short_limit_slidder,
                              relevant_token_limit=self.long_limit_slidder,
                              relevance_score_threshold=self.rel_score_threshold_slidder,
                              similarity_score_threshold=self.sim_score_threshold_slidder))
        
    def knowledge_base_settings(self) -> None:
        """Create settings for memory.
        """
        self.kb_limit_slider = st.slider('Knowledge base token limit', min_value=0, max_value=6000, step=1, 
                value=self.backend.knowledge_base_config['kb_token_limit'], disabled=self.generating)
        self.kb_score_threshold_slider = st.slider('Relevance score threshold for knowledge base', min_value=0.0, max_value=1.0, step=0.01,
                value=self.backend.knowledge_base_config['kb_score_threshold'], disabled=self.generating)
        summary = [
            'Current settings:',
            f"Short term memory token limit: {self.backend.knowledge_base_config['kb_token_limit']}",
            f"Relevance score threshold: {self.backend.knowledge_base_config['kb_score_threshold']}"
        ]
        st.markdown('  \n'.join(summary))
        st.button(label=':floppy_disk:', key='kb_token_save', disabled=self.generating, 
                  use_container_width=True,
                  on_click=self.backend.set_knowledge_base_config,
                  kwargs=dict(kb_token_limit=self.kb_limit_slider,
                              kb_score_threshold=self.kb_score_threshold_slider))

    def model_settings(self) -> None:
        """Create settings for text generation.
        """
        self.temperature_slider = st.slider('Temparature', min_value=0.0, max_value=2.0, step=0.01, value=self.backend.generation_config['temperature'], disabled=self.generating)
        self.max_new_token_slider = st.slider('Maximum number of new tokens', min_value=0, max_value=4096, step=1, value=self.backend.generation_config['max_new_tokens'], disabled=self.generating)
        self.repetition_slider = st.slider('Repetition penalty', min_value=1.0, max_value=2.0, step=0.01, value=self.backend.generation_config['repetition_penalty'], disabled=self.generating)
        self.topp_slider = st.slider('Top P', min_value=0.0, max_value=1.0, step=0.01, value=self.backend.generation_config['top_p'], disabled=self.generating)
        self.topk_slider = st.slider('Top K', min_value=0, max_value=30000, step=1, value=self.backend.generation_config['top_k'], disabled=self.generating)
        summary = [
            'Current settings:',
            f"Temperature: {self.backend.generation_config['temperature']}",
            f"Max new tokens: {self.backend.generation_config['max_new_tokens']}",
            f"Repetition penalty: {self.backend.generation_config['repetition_penalty']}",
            f"Top P: {self.backend.generation_config['top_p']}",
            f"Top K: {self.backend.generation_config['top_k']}",
        ]
        st.markdown('  \n'.join(summary))
        st.button(label=':floppy_disk:', 
                  key='llm_config_save', 
                  disabled=self.generating, 
                  use_container_width=True,
                  on_click=self.backend.set_generation_config,
                  kwargs=dict(temperature=self.temperature_slider,
                              max_new_tokens=self.max_new_token_slider,
                              repetition_penalty=self.repetition_slider,
                              top_p=self.topp_slider,
                              top_k=self.topk_slider))

    def tool_settings(self) -> None:
        """Create settings for tools.
        """
        for k, v in self.backend.tool_status.items():
            pretty_name = k.replace('_', ' ').strip().title()
            st.toggle(label=f'{pretty_name}', value=v, on_change=self.backend.toggle_tool, kwargs=dict(tool_name=k),
                        disabled=self.generating)

    def sidebar(self) -> None:
        """Creating the sidebar.
        """
        with st.sidebar:
            app_summary = ['Powered by:', 
                        f'* LLM: {self.backend.factory.model_id}', 
                        f'* Embedding model: {self.backend.embeddings.name}', 
                        '', 
                        'Current conversation:', 
                        f'* {self.backend.memory.title}',
                        '',
                        'Current prompt format:',
                        f'* {self.backend.prompt_template.template_name}']
            st.header(PACKAGE_DISPLAY_NAME.upper(), help='  \n'.join(app_summary), divider="grey")
            with st.expander(label=':left_speech_bubble: Conversations', expanded=True):
                self.chats()

            with st.expander(label=':paperclip: Knowledge Bases', expanded=False):
                self.knowledge_base_config()

            with st.expander(label=':gear: Settings', expanded=False):
                self.settings()

    # Chatbot frontend
    def create_input_config(self, user_input: str, begin_text: str, generation_mode: Literal['new', 'retry', 'continue']) -> None:
        """Create everything needed for the next generation.

        Args:
            user_input (str): User request.
            begin_text (str): Starting text of the response.
            generation_mode (Literal[&#39;new&#39;, &#39;retry&#39;, &#39;continue&#39;]): Mode of generation.
        """
        new_input = user_input.strip()
        ai_start = begin_text.lstrip()
        if generation_mode != 'new':
            last_record = self.backend.memory.history[-1]
            new_input = last_record[0]
            if generation_mode == 'continue':
                ai_start = last_record[1]
            self.backend.memory.remove_last_interaction()
        prompt_args = dict(
            llm=self.backend.llm,
            memory=self.backend.memory,
            user_input=new_input,
            prompt_template=self.backend.prompt_template,
            system=self.backend.memory.system,
            recent_token_limit=self.backend.memory_config['recent_token_limit'],
            relevant_token_limit=self.backend.memory_config['relevant_token_limit'],
            relevance_score_threshold=self.backend.memory_config['relevance_score_threshold'],
            similarity_score_threshold=self.backend.memory_config['similarity_score_threshold'],
            knowledge_base=self.backend.knowledge_base,
            kb_token_limit=self.backend.knowledge_base_config['kb_token_limit'],
            kb_score_threshold=self.backend.knowledge_base_config['kb_score_threshold']
        )
        if ai_start.strip() != '':
            prompt = create_prompt_with_history(**prompt_args) + ai_start
            st.session_state.begin_text_cache = ai_start
        elif ((not self.backend.tool_selector.is_empty) & (self.backend.prompt_template.allow_custom_role)):
            prompt_args['tool_selector'] = self.backend.tool_selector
            prompt = create_prompt_with_history(**prompt_args)
        else:
            prompt = create_prompt_with_history(**prompt_args)
        
        st.session_state.input_config = dict(
            prompt=prompt,
            user_input=new_input,
            begin_text=ai_start,
            mode=generation_mode
        )

    def process_image(self, messages: List[Dict[str, Any]], tool_output: Dict[str, Any]) -> str:
        """Capture images in tool output.

        Args:
            messages (List[Dict[str, Any]]): List of messages to form the prompt.
            tool_output (Dict[str, Any]): Output of the tool.

        Returns:
            str: Prompt for generation.
        """
        if tool_output.get('output', dict()).get('images') is not None:
            images = tool_output.get('output', dict()).get('images')
            begin_text = ''
            img_dir = []
            try:
                for i, img in enumerate(images):
                    try:
                        st.image(img)
                        img_dir.append(f'![Image {i}]({img})')
                    except:
                        pass
            except:
                pass
            if img_dir:
                begin_text = '  \n'.join(img_dir) + '\n\n'
                prompt = self.backend.prompt_template.create_custom_prompt(messages=messages) + begin_text
                return prompt
            else:
                return self.backend.prompt_template.create_custom_prompt(messages=messages)
        else:
            return self.backend.prompt_template.create_custom_prompt(messages=messages)
        
    def process_footnote(self, tool_output: Dict[str, Any]) -> str:
        """Check if footnote exist in the tool output.

        Args:
            tool_output (Dict[str, Any]): Tool output dictionary.

        Returns:
            str: If footnote exist, return the footnote string, otherwise return a empty string.
        """
        footnote = tool_output.get('output', dict()).get('footnote')
        if footnote:
            output = '\n\n---\n'
            if isinstance(footnote, str):
                return output + footnote
            elif isinstance(footnote, list):
                for i in footnote:
                    output += str(i) + '  \n'
                return output.rstrip()
            else:
                return output + str(footnote)
        else:
            return ''

    def generation_message(self) -> None:
        """Create streaming message.
        """
        import json
        if self.input_config is not None:
            import os
            from ..utils import current_time, save_json
            logfilename = os.path.join(self.log_dir, f'{self.backend.memory.chat_id}_{current_time()}.json')
            save_json(content=self.input_config, file_dir=logfilename)
            with st.chat_message(name='user'):
                st.markdown(self.input_config['user_input'])
            with st.chat_message(name='assistant'):
                with st.spinner('Thinking....'):
                    prompt = self.input_config['prompt']
                    begin_text = self.input_config['begin_text']
                    save_args = dict()
                    footnote = ''
                    if not isinstance(prompt, str):
                        tool_input = self.backend.tool_selector.tool_call_input(llm=self.backend.llm, messages=prompt, prompt_template=self.backend.prompt_template, **self.backend.generation_config)
                        if tool_input['name'] == 'direct_response':
                            prompt = self.backend.prompt_template.create_custom_prompt(messages=prompt) + begin_text
                        else:
                            toolholder = st.empty()
                            tool_name = tool_input['name'].replace('_', ' ').title().strip()
                            with toolholder.status(label=f":hammer_and_pick: Running __{tool_name}__...", state='running'):
                                st.text(json.dumps(tool_input, indent=4))
                                tool_output = self.backend.tool_selector.tool_call_output(tool_input=tool_input, return_error=True)
                                save_args['function_call'] = tool_output
                            with toolholder.status(label=f":hammer_and_pick: __{tool_name}__", state='complete'):
                                st.text(json.dumps(tool_output, indent=4))
                            prompt.append(dict(role='function_call', content=str(tool_output)))
                            prompt = self.process_image(messages=prompt, tool_output=tool_output) + begin_text
                            footnote = self.process_footnote(tool_output=tool_output)

                    placeholder = st.empty()
                    streamer = self.backend.llm.stream(prompt, stop=self.backend.prompt_template.stop, **self.backend.generation_config)
                    output = begin_text
                    placeholder.markdown(output.strip(' \r\n\t'))
                    for i in streamer:
                        output += i
                        placeholder.markdown(output.strip(' \r\n\t'))
                    if footnote != '':
                        placeholder.markdown(output.strip(' \r\n\t') + footnote)
                    output = output.strip()
                    self.backend.memory.save_interaction(user_input=self.input_config['user_input'], assistant_output=output, **save_args)

                    # Change title if the title is New Chat
                    if self.backend.memory.title == 'New Chat':
                        pref = '\n\nNote: This is the short title of this conversation: {"title": '
                        title = gen_string(self.backend.llm, prompt=prompt + output + pref, max_new_tokens=30)
                        if title != 'New Chat':
                            self.backend.memory.update_title(title)
                    st.session_state.input_config = None
                    st.session_state.generating = False
                    st.rerun()

    def historical_conversation(self) -> None:
        """Creating the historical conversation.
        """
        import json
        history = self.backend.memory.history_dict
        for message in history:
            with st.chat_message(name=message['role']):
                fn_call = message.get('function_call')
                content = message['content']
                if fn_call is not None:
                    fn_name = fn_call['name'].replace('_', ' ').strip().title()
                    with st.status(label=f":hammer_and_pick: __{fn_name}__", state='complete'):
                        st.code(json.dumps(fn_call, indent=4), language='plaintext')
                    if fn_call.get('output', dict()).get('footnote') is not None:
                        content += self.process_footnote(fn_call)
                    if fn_call.get('output', dict()).get('images') is not None:
                        images = fn_call.get('output', dict()).get('images')
                        if isinstance(images, list):
                            for img in images:
                                try:
                                    st.image(img)
                                except:
                                    pass
                st.markdown(content, help=f'Number of tokens: {self.backend.llm.get_num_tokens(message["content"])}')
        self.generation_message()

    def util_buttons(self) -> None:
        """Create utility buttons for text generation.
        """
        def retry_response(mode):
            if self.backend.memory.interaction_count != 0:
                self.create_input_config(user_input='sample', begin_text=self.begin_text, generation_mode=mode)
                st.session_state.generating = True

        btns = row([1, 1, 1])
        btns.button(':arrows_counterclockwise:', use_container_width=True, help='Re-generate response', disabled=self.generating, 
                      on_click=retry_response, kwargs=dict(mode='retry'))

        btns.button(':fast_forward:', use_container_width=True, help='Continue generating response', disabled=self.generating,
                      on_click=retry_response, kwargs=dict(mode='retry'))

        btns.button(':wastebasket:', use_container_width=True, help='Remove the latest question and response', disabled=self.generating, 
                       on_click=self.backend.memory.remove_last_interaction)

    def input_box(self) -> None:
        """Creating the input box.
        """
        with bottom():
            self.user_input = st.chat_input(placeholder='Your message...', disabled=self.generating)
            if self.enable_begin_text:
                self.begin_text = st.text_area(label='response_start', placeholder='Starting text of the response here...', value=self.begin_text_cache, label_visibility='collapsed')
            else:
                self.begin_text = ''

            if self.user_input:
                if self.user_input.strip() != '':
                    self.create_input_config(user_input=self.user_input, begin_text=self.begin_text, generation_mode='new')
                    st.session_state.generating = True
                    st.rerun()

            if self.mobile:
                with st.expander(':gear: Extra options'):
                    self.util_buttons()
            else:
                self.util_buttons()
            
    def chatbot(self) -> None:
        """Creating the chatbot interface.
        """
        if self.backend.knowledge_base is not None:
            title = self.backend.knowledge_base_map[self.backend.knowledge_base.kb_id]["title"].replace("_", " ").title()
            files = 'Files:  \n' + '  \n'.join(list(map(lambda x: x[0], self.backend.knowledge_base.files)))
            st.button(f'__{title}__ :heavy_multiplication_x:', on_click=self.backend.detach_knowledge_base, help=files)
        self.historical_conversation()
        self.input_box()

    def run(self) -> None:
        """Run the app.
        """
        if self.is_login:
            self.sidebar()
            self.chatbot()
        else:
            self.login()
