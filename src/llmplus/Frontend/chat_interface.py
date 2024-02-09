import os
from ..Models.Factory.llm_factory import LlmFactory
from ..Embeddings.base_embeddings import BaseEmbeddingsToolkit
from typing import Dict, Any, Union, List, Tuple, Type


class ChatInterface:

    def __init__(self, model: LlmFactory, embeddings: Type[BaseEmbeddingsToolkit]) -> None:
        from ..Memory.long_short_memory import LongShortTermChatMemory
        from ..Prompts.prompt_template import DEFAULT_SYSTEM_MESSAGE, PromptTemplate
        self.model = model
        self.embeddings = embeddings
        self.memory = LongShortTermChatMemory(title='Untitled 0', embeddings=self.embeddings, from_exist=False)
        self.system = DEFAULT_SYSTEM_MESSAGE
        self.template = self.model.prompt_template
        self.llm = self.model(stop=self.template.stop + ['###'])
        self.short_limit = 600
        self.long_limit = 500
        self.score_threshold = 0.5
        self.mobile = False
        self.bot = dict()
        self.state = dict(generate=False)

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

    @property
    def config_dict(self) -> Dict[str, Any]:
        from ..Prompts.prompt_template import DEFAULT_SYSTEM_MESSAGE
        config = dict(
            # conversation list
            new = dict(obj='textbox', args=dict(value = '', interactive=True, show_label=False, placeholder='New chat title here...', scale=4, container=False)),
            add = dict(obj='btn', args=dict(value='Add', min_width=15, variant='primary', size='sm', scale=1, interactive=True)),
            convos = dict(obj='dropdown', args=dict(choices=self.titles, label='Conversations', value=None, container=True)),
            select = dict(obj='btn', args=dict(value='Select', min_width=25, variant='primary', size='sm', interactive=True)),
            remove = dict(obj='btn', args=dict(value='Remove', min_width=25, variant='secondary', size='sm', interactive=True)),

            # main console
            bot = dict(obj='chatbot', args=dict(label=self.current_title, height=600, value=self.history, show_copy_button=True)),
            user = dict(obj='textbox', args=dict(value = '', interactive=True, show_label=False, placeholder='Type your message here...', max_lines=40, scale=7, container=False)),
            send = dict(obj='btn', args=dict(value='Send', min_width=35, variant='primary', size='sm', scale=2, interactive=True)),
            cont = dict(obj='btn', args=dict(value='Continue', min_width=15, size='sm', scale=1, interactive=True)),
            start = dict(obj='textbox', args=dict(value = '', interactive=True, show_label=False, placeholder='Type the start of the chatbot message...', max_lines=40, scale=7, container=False)),
            regen = dict(obj='btn', args=dict(value='Retry', min_width=15, size='sm', scale=2, interactive=True)),
            rmlast = dict(obj='btn', args=dict(value='Undo', min_width=15, size='sm', scale=2, interactive=True)),

            # settings
            # system settings 
            system = dict(obj='textbox', args=dict(value=DEFAULT_SYSTEM_MESSAGE, show_label=False, placeholder='System message here...', lines=8, max_lines=40, scale=4, interactive=True)),
            long_limit = dict(obj='slider', args=dict(value=self.long_limit, minimum=0, maximum=4000, step=1, label='Long term memory tokens limit')),
            short_limit = dict(obj='slider', args=dict(value=self.short_limit, minimum=0, maximum=6000, step=1, label='Short term memory tokens limit')),
            sim_score = dict(obj='slider', args=dict(value=self.score_threshold, minimum=0, maximum=1, step=0.01, label='Long term memory relevance score threshold')),
            sys_save = dict(obj='btn', args=dict(value='Save', min_width=20, variant='secondary', size='sm', scale=2, interactive=True)),
            sys_log = dict(obj='text', args=dict(show_label=False, value=self.get_memory_settings(), lines=5, container=False)),

            # format settings
            templates = dict(obj='dropdown', args=dict(choices=self.presets, value=self.template.template_name, show_label=False, interactive=True, container=False)),
            format_save = dict(obj='btn', args=dict(value='Save', min_width=20, variant='secondary', size='sm', scale=2, interactive=True)),
            format_log = dict(obj='text', args=dict(show_label=False, value=self.get_prompt_settings(), lines=2, container=False)),

            # LLM settings
            temperature = dict(obj='slider', args=dict(value=self.llm.generation_config['temperature'], minimum=0, maximum=2, step=0.01, label='Temperature')),
            tokens = dict(obj='slider', args=dict(value=self.llm.generation_config['max_new_tokens'], minimum=0, maximum=4096, step=1, label='Maximum number of new tokens')),
            repeat = dict(obj='slider', args=dict(value=self.llm.generation_config['repetition_penalty'], minimum=0, maximum=2, step=0.01, label='Repetition penalty')),
            topp = dict(obj='slider', args=dict(value=self.llm.generation_config['top_p'], minimum=0, maximum=1, step=0.001, label='Top P')),
            topk = dict(obj='slider', args=dict(value=self.llm.generation_config['top_k'], minimum=0, maximum=500, step=1, label='Top K')),
            llm_save = dict(obj='btn', args=dict(value='Save', min_width=20, variant='secondary', size='sm', scale=2, interactive=True)),
            llm_log = dict(obj='text', args=dict(show_label=False, value=self.get_llm_settings(), lines=6, container=False)),
        )
        return config
    
    @property
    def mobile_config_dict(self) -> Dict[str, Any]:
        from copy import deepcopy
        config = deepcopy(self.config_dict)
        changes = dict(
            bot = dict(height=450, show_copy_button=False),
            start = dict(scale=4, placeholder='Start of bot message...'),
            send = dict(scale=1),
            cont = dict(scale=2)
        )
        for k, v in changes.items():
            for a, i in v.items():
                config[k]['args'][a] = i
        return config
    
    @property
    def buttons(self) -> List[str]:
        """List of buttons except send.

        Returns:
            List[str]: List of buttons except send.
        """
        btns = []
        for k, v in self.config_dict.items():
            if ((v['obj'] == 'btn') & (k != 'send')):
                btns.append(k)
        return btns

    def get_memory_settings(self) -> str:
        settings = [
            'Current memory settings:',
            f'System message size: {self.llm.get_num_tokens(self.system)} tokens',
            f'Long term memory tokens limit: {self.long_limit}',
            f'Short term memory tokens limit: {self.short_limit}',
            f'Long term memory relevance score threshold: {self.score_threshold}'
        ]
        return '\n'.join(settings)
    
    def get_prompt_settings(self) -> str:
        settings = [
            'Current prompt settings:',
            f'Preset: {self.template.template_name}'
        ]
        return '\n'.join(settings)
    
    def get_llm_settings(self) -> str:
        settings = [
            'Current LLM settings:',
            f'Temperature: {self.llm.generation_config["temperature"]}',
            f'Maximum number of new tokens: {self.llm.generation_config["max_new_tokens"]}',
            f'Repetition penalty: {self.llm.generation_config["repetition_penalty"]}',
            f'Top P: {self.llm.generation_config["top_p"]}',
            f'Top K: {self.llm.generation_config["top_k"]}'
        ]
        return '\n'.join(settings)

    def get_prompt(self, user_input: str, ai_start: str = '') -> str:
        from ..Memory.long_short_memory import create_long_short_prompt
        prompt = create_long_short_prompt(
            user=user_input,
            prompt_template=self.template,
            llm=self.llm,
            memory=self.memory,
            system=self.system,
            short_token_limit=self.short_limit,
            long_token_limit=self.long_limit,
            score_threshold=self.score_threshold
        ) + ai_start.lstrip(' \r\n\t')
        return prompt

    def change_prompt_format(self, template: str) -> str:
        """Changing the prompt format.

        Args:
            template (str): Preset name from the dropdown menu.
        """
        from ..Prompts.prompt_template import PromptTemplate
        self.template = PromptTemplate.from_preset(template)
        return self.get_prompt_settings()

    def change_memory_setting(self, system: str, long_limit: int, short_limit: int, sim_score: float) -> str:
        """Changing the system message and memory settings.

        Args:
            system (str): System textbox.
            long_limit (int): Long term memory slider.
            short_limit (int): Short term memory slider.
            sim_score (float): Similarity score threshold slider.
        """
        self.system = system.strip(' \n\r\t')
        self.memory.vectordb._info['system'] = self.system
        self.memory.save()
        self.long_limit = long_limit
        self.short_limit = short_limit
        self.score_threshold = sim_score
        return self.get_memory_settings()

    def change_llm_setting(self, temperature: float, max_tokens: int, repeat_penalty: float, top_p: float, top_k: int) -> str:
        """Change llm generation settings.

        Args:
            temperature (float): Temperature of the llm.
            max_tokens (int): Maximum number of tokens to generate.
            repeat_penalty (float): Repetition penalty.
            top_p (float): Top P.
            top_k (int): Top K.
        """
        self.llm = self.model(
            temperature=temperature,
            repetition_penalty=repeat_penalty,
            max_new_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=self.template.stop + ['###']
            )
        return self.get_llm_settings()

    def change_memory(self, btn: str, title: str) -> Tuple[Any]:
        """Handling memory changing settings.

        Args:
            btn (str): Button that triggered this function.
            title (str): Title used for this trigger.

        Returns:
            Tuple[Any]: New title textbox, Chats dropdown menu, the Chatbot box, and the system textbox.
        """
        import gradio as gr
        from ..Memory.long_short_memory import LongShortTermChatMemory
        from ..Prompts.prompt_template import DEFAULT_SYSTEM_MESSAGE
        title = title.strip(' \r\n\t')
        if ((btn == 'Add') & (title != '') & (title not in self.titles)):
            self.memory = LongShortTermChatMemory(title=title, embeddings=self.embeddings, from_exist=True)
        elif ((btn == 'Select') & (title != '') & (title is not None)):
            from_exist = title != 'Untitled 0'
            self.memory = LongShortTermChatMemory(title=title, embeddings=self.embeddings, from_exist=from_exist)
        elif ((btn == 'Remove') & (title != '') & (title is not None)):
            import shutil
            if title ==self.current_title:
                shutil.rmtree(self.memory.chat_dir)
                self.memory =  LongShortTermChatMemory(title='Untitled 0', embeddings=self.embeddings, from_exist=False)
            else:
                mem = LongShortTermChatMemory(title=title, embeddings=self.embeddings, from_exist=False)
                shutil.rmtree(mem.chat_dir)
        system = self.memory.info.get('system', DEFAULT_SYSTEM_MESSAGE)
        self.system = system
        # Things to return: new, title dropdown, chatbot, system
        return gr.Textbox(value=''), gr.Dropdown(choices=self.titles, value=None), gr.Chatbot(value=self.history, label=self.current_title), gr.Textbox(value=system, interactive=True), self.get_memory_settings()

    def remove_last(self) -> List[List[str]]:
        """Removing the last interaction of the conversation.

        Returns:
            List[List[str]]: Conversation history after removing the last interaction.
        """
        if len(self.history) != 0:
            self.memory.remove_last_interaction()
        return self.history

    def input_handler(self, btn: str, user: str, start: str, bot: List[List[str]]) -> Tuple[Any]:
        """Handling GUI and class attributes before text generation.

        Args:
            btn (str): Button used to trigger the function.
            user (str): User input.
            start (str): Start of the chatbot output, should be an empty string by default.
            bot (List[List[str]]): Chatbot conversation history.

        Returns:
            Tuple[Any]: send button, user input box, conversation box, and all other buttons.
        """
        import gradio as gr
        user_input = user.strip(' \n\r\t')
        freeze = True
        if ((btn == 'Send') & (user != '')):
            self.state['generate'] = True
            self.state['prompt'] = self.get_prompt(user_input=user_input, ai_start=start)
            self.state['start'] = start
            user = ''
            bot = self.history + [[user_input, None]]
        elif((btn == 'Retry') & (len(self.history) != 0)):
            user_input = self.history[-1][0]
            self.memory.remove_last_interaction()
            self.state['generate'] = True
            self.state['prompt'] = self.get_prompt(user_input=user_input, ai_start=start)
            self.state['start'] = start
            bot = self.history + [[user_input, None]]
        elif((btn == 'Continue') & (len(self.history) != 0)):
            user_input = self.history[-1][0]
            ai_start = self.history[-1][1]
            self.memory.remove_last_interaction()
            self.state['generate'] = True
            self.state['prompt'] = self.get_prompt(user_input=user_input, ai_start=ai_start)
            self.state['start'] = ai_start
            bot = self.history + [[user_input, None]]
        elif btn == 'Stop':
            self.state = dict(generate=False)
            bot = bot
        else:
            freeze = False

        if freeze:
            returns = [
                gr.Button(value='Stop', variant='stop'),
                user,
                bot,
            ] + [gr.Button(interactive=False)] * len(self.buttons)
        else:
            returns = [
                gr.Button(value='Send'),
                user,
                self.history,
            ] + [gr.Button(interactive=True)] * len(self.buttons)
        return tuple(returns)

    def generation(self, bot: List[List[str]]) -> List[List[str]]:
        """Text generation.

        Args:
            bot (List[List[str]]): Chatbot conversation history.

        Returns:
            List[List[str]]: The updated conversation.
        """
        if self.state['generate']:
            output = self.state['start']
            prompt = self.state['prompt']
            print(f'Input tokens: {self.llm.get_num_tokens(prompt)}')
            for i in self.llm.stream(prompt):
                if self.state['generate']:
                    output += i
                    bot[-1][1] = output.strip(' \r\n\t')
                    yield bot
                else:
                    yield bot
                    break
            if bot[-1][1] is None:
                bot[-1][1] = ''
            self.memory.save_interaction(bot[-1][0], bot[-1][1])
            self.state = dict(generate=False)
            yield bot
        else:
            yield bot

    def postgen_handler(self) -> List[Any]:
        """Reactivate all the buttons.

        Returns:
            List[Any]: All buttons.
        """
        import gradio as gr
        return [gr.Button(value='Send', variant='primary', interactive=True)] + ([gr.Button(interactive=True)] * len(self.buttons))

    def vars(self, var_name: str, **kwargs: Dict[str, Any]) -> Any:
        """Generate the gradio component in the config dictionary given the key in the config dict.

        Args:
            var_name (str): Key in the config dict.

        Returns:
            Any: The gradio component.
        """
        from copy import deepcopy
        import gradio as gr
        keys = ['btn', 'text', 'textbox', 'dropdown', 'chatbot', 'slider']
        values = [gr.Button, gr.Text, gr.Textbox, gr.Dropdown, gr.Chatbot, gr.Slider]
        type_map = dict(zip(keys, values))
        vars = deepcopy(self.mobile_config_dict if self.mobile else self.config_dict)

        output = vars[var_name].copy()
        for k, v in kwargs.items():
            output['args'][k] = v
        output = type_map[output['obj']](**output['args'])
        return output
    
    def output_map(self, keys: Union[str, List[str]]) -> List[Any]:
        if type(keys) == str:
            keys = [keys]
        return list(map(lambda x: self.bot[x], keys))
    
    def _init_pc_frame(self) -> None:
        import gradio as gr
        with gr.Blocks() as self.frame:
            with gr.Tab(label='Chat'):
                with gr.Row():
                    with gr.Column(): # Conversations
                        with gr.Group():
                            with gr.Row(variant='compact'):
                                self.bot['new'] = self.vars('new')
                                self.bot['add'] = self.vars('add')
                            self.bot['convos'] = self.vars('convos')
                            with gr.Row(variant='compact'):
                                self.bot['select'] = self.vars('select')
                                self.bot['remove'] = self.vars('remove')
                        with gr.Accordion(label='Prompt format settings', open=True):
                            self.bot['templates'] = self.vars('templates')
                            self.bot['format_save'] = self.vars('format_save')
                            self.bot['format_log'] = self.vars('format_log')

                    with gr.Column(scale=4): # Main console
                        self.bot['bot'] = self.vars('bot')
                        with gr.Row(variant='compact'):
                            self.bot['user'] = self.vars('user')
                            self.bot['send'] = self.vars('send')
                            self.bot['cont'] = self.vars('cont')
                        with gr.Row(variant='compact'):
                            self.bot['start'] = self.vars('start')
                            self.bot['regen'] = self.vars('regen')
                            self.bot['rmlast'] = self.vars('rmlast')

            with gr.Tab(label='Settings'): # Settings
                with gr.Row():
                    with gr.Column(scale=1):
                        self.bot['system'] = self.vars('system')
                        self.bot['long_limit'] = self.vars('long_limit')
                        self.bot['short_limit'] = self.vars('short_limit')
                        self.bot['sim_score'] = self.vars('sim_score')
                        self.bot['sys_save'] = self.vars('sys_save')
                        self.bot['sys_log'] = self.vars('sys_log')
                    with gr.Column(scale=1):
                        self.bot['temperature'] = self.vars('temperature')
                        self.bot['tokens'] = self.vars('tokens')
                        self.bot['repeat'] = self.vars('repeat')
                        self.bot['topp'] = self.vars('topp')
                        self.bot['topk'] = self.vars('topk')
                        self.bot['llm_save'] = self.vars('llm_save')
                        self.bot['llm_log'] = self.vars('llm_log')


            # functions
            self.bot['add'].click(fn=self.change_memory, inputs=self.output_map(['add', 'new']), outputs=self.output_map(['new', 'convos', 'bot', 'system', 'sys_log']))
            self.bot['select'].click(fn=self.change_memory, inputs=self.output_map(['select', 'convos']), outputs=self.output_map(['new', 'convos', 'bot', 'system', 'sys_log']))
            self.bot['remove'].click(fn=self.change_memory, inputs=self.output_map(['remove', 'convos']), outputs=self.output_map(['new', 'convos', 'bot', 'system', 'sys_log']))

            self.bot['format_save'].click(fn=self.change_prompt_format, inputs=self.output_map(['templates']), outputs=self.output_map(['format_log']))
            self.bot['sys_save'].click(fn=self.change_memory_setting, inputs=self.output_map(['system', 'long_limit', 'short_limit', 'sim_score']), outputs=self.output_map(['sys_log']))
            self.bot['llm_save'].click(fn=self.change_llm_setting, inputs=self.output_map(['temperature', 'tokens', 'repeat', 'topp', 'topk']), outputs=self.output_map(['llm_log']))
            self.bot['rmlast'].click(fn=self.remove_last, outputs=self.output_map('bot'))

            self.bot['send'].click(fn=self.input_handler, 
                                   inputs=self.output_map([ 'send', 'user', 'start', 'bot']), 
                                   outputs=self.output_map(['send', 'user', 'bot'] + self.buttons),
                                   queue=False).then(
                                       fn=self.generation, inputs=self.output_map('bot'), outputs=self.output_map('bot')
                                   ).then(
                                       fn=self.postgen_handler, outputs=self.output_map(['send'] + self.buttons)
                                   )
            self.bot['user'].submit(fn=self.input_handler, 
                                   inputs=self.output_map([ 'send', 'user', 'start', 'bot']), 
                                   outputs=self.output_map(['send', 'user', 'bot'] + self.buttons),
                                   queue=False).then(
                                       fn=self.generation, inputs=self.output_map('bot'), outputs=self.output_map('bot')
                                   ).then(
                                       fn=self.postgen_handler, outputs=self.output_map(['send'] + self.buttons)
                                   )
            self.bot['cont'].click(fn=self.input_handler, 
                                   inputs=self.output_map([ 'cont', 'user', 'start', 'bot']), 
                                   outputs=self.output_map(['send', 'user', 'bot'] + self.buttons),
                                   queue=False).then(
                                       fn=self.generation, inputs=self.output_map('bot'), outputs=self.output_map('bot')
                                   ).then(
                                       fn=self.postgen_handler, outputs=self.output_map(['send'] + self.buttons)
                                   )
            self.bot['regen'].click(fn=self.input_handler, 
                                   inputs=self.output_map([ 'regen', 'user', 'start', 'bot']), 
                                   outputs=self.output_map(['send', 'user', 'bot'] + self.buttons),
                                   queue=False).then(
                                       fn=self.generation, inputs=self.output_map('bot'), outputs=self.output_map('bot')
                                   ).then(
                                       fn=self.postgen_handler, outputs=self.output_map(['send'] + self.buttons)
                                   )

    def _init_mobile_frame(self) -> None:
        import gradio as gr
        with gr.Blocks() as self.frame:
            with gr.Tab(label='Chat'):
                self.bot['bot'] = self.vars('bot')
                self.bot['user'] = self.vars('user')
                with gr.Row('compact'):
                    self.bot['start'] = self.vars('start')
                    self.bot['send'] = self.vars('send')
                with gr.Row(variant='compact'):
                    self.bot['cont'] = self.vars('cont')
                    self.bot['regen'] = self.vars('regen')
                    self.bot['rmlast'] = self.vars('rmlast')

            with gr.Tab(label='Conversations'):
                with gr.Group():
                    with gr.Row(variant='compact'):
                        self.bot['new'] = self.vars('new')
                        self.bot['add'] = self.vars('add')
                    self.bot['convos'] = self.vars('convos')
                    with gr.Row(variant='compact'):
                        self.bot['select'] = self.vars('select')
                        self.bot['remove'] = self.vars('remove')
                with gr.Accordion(label='Prompt format settings', open=True):
                    self.bot['templates'] = self.vars('templates')
                    self.bot['format_save'] = self.vars('format_save')
                    self.bot['format_log'] = self.vars('format_log')


            with gr.Tab(label='Settings'): # Settings
                with gr.Accordion('System and memory settings', open=True):
                    self.bot['system'] = self.vars('system')
                    self.bot['long_limit'] = self.vars('long_limit')
                    self.bot['short_limit'] = self.vars('short_limit')
                    self.bot['sim_score'] = self.vars('sim_score')
                    self.bot['sys_save'] = self.vars('sys_save')
                    self.bot['sys_log'] = self.vars('sys_log')
                with gr.Accordion('LLM settings', open=True):
                    self.bot['temperature'] = self.vars('temperature')
                    self.bot['tokens'] = self.vars('tokens')
                    self.bot['repeat'] = self.vars('repeat')
                    self.bot['topp'] = self.vars('topp')
                    self.bot['topk'] = self.vars('topk')
                    self.bot['llm_save'] = self.vars('llm_save')
                    self.bot['llm_log'] = self.vars('llm_log')


            # functions
            self.bot['add'].click(fn=self.change_memory, inputs=self.output_map(['add', 'new']), outputs=self.output_map(['new', 'convos', 'bot', 'system', 'sys_log']))
            self.bot['select'].click(fn=self.change_memory, inputs=self.output_map(['select', 'convos']), outputs=self.output_map(['new', 'convos', 'bot', 'system', 'sys_log']))
            self.bot['remove'].click(fn=self.change_memory, inputs=self.output_map(['remove', 'convos']), outputs=self.output_map(['new', 'convos', 'bot', 'system', 'sys_log']))

            self.bot['format_save'].click(fn=self.change_prompt_format, inputs=self.output_map(['templates']), outputs=self.output_map(['format_log']))
            self.bot['sys_save'].click(fn=self.change_memory_setting, inputs=self.output_map(['system', 'long_limit', 'short_limit', 'sim_score']), outputs=self.output_map(['sys_log']))
            self.bot['llm_save'].click(fn=self.change_llm_setting, inputs=self.output_map(['temperature', 'tokens', 'repeat', 'topp', 'topk']), outputs=self.output_map(['llm_log']))
            self.bot['rmlast'].click(fn=self.remove_last, outputs=self.output_map('bot'))

            self.bot['send'].click(fn=self.input_handler, 
                                   inputs=self.output_map([ 'send', 'user', 'start', 'bot']), 
                                   outputs=self.output_map(['send', 'user', 'bot'] + self.buttons),
                                   queue=False).then(
                                       fn=self.generation, inputs=self.output_map('bot'), outputs=self.output_map('bot')
                                   ).then(
                                       fn=self.postgen_handler, outputs=self.output_map(['send'] + self.buttons)
                                   )
            self.bot['cont'].click(fn=self.input_handler, 
                                   inputs=self.output_map([ 'cont', 'user', 'start', 'bot']), 
                                   outputs=self.output_map(['send', 'user', 'bot'] + self.buttons),
                                   queue=False).then(
                                       fn=self.generation, inputs=self.output_map('bot'), outputs=self.output_map('bot')
                                   ).then(
                                       fn=self.postgen_handler, outputs=self.output_map(['send'] + self.buttons)
                                   )
            self.bot['regen'].click(fn=self.input_handler, 
                                   inputs=self.output_map([ 'regen', 'user', 'start', 'bot']), 
                                   outputs=self.output_map(['send', 'user', 'bot'] + self.buttons),
                                   queue=False).then(
                                       fn=self.generation, inputs=self.output_map('bot'), outputs=self.output_map('bot')
                                   ).then(
                                       fn=self.postgen_handler, outputs=self.output_map(['send'] + self.buttons)
                                   )



    def launch(self, mobile: bool = False, **kwargs) -> None:
        if mobile:
            self.mobile = True
            default_config = dict(inbrowser=False, share=False, auth=None)
            default_config.update(kwargs)
            self._init_mobile_frame()
        else:
            default_config = dict(inbrowser=True, share=False, auth=None)
            default_config.update(kwargs)
            self._init_pc_frame()
        self.frame.launch(**default_config)

