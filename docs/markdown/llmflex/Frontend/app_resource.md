Module llmflex.Frontend.app_resource
====================================

Classes
-------

`AppBackend(config: Dict[str, Any])`
:   Resources for the App to share.
        
    
    Initialise the backend resourses.
    Args:
        config (Dict[str, Any]): Configuration of all the resources.

    ### Instance variables

    `config: Dict[str, Any]`
    :   Configuration of all the resources.ry_
        
        Returns:
            Dict[str, Any]: Configuration of all the resources.

    `embeddings: llmflex.Embeddings.base_embeddings.BaseEmbeddingsToolkit`
    :   Embeddings toolkit.
        
        Returns:
            BaseEmbeddingsToolkit: Embeddings toolkit.

    `factory: llmflex.Models.Factory.llm_factory.LlmFactory`
    :   LLM factory.
        
        Returns:
            LlmFactory: LLM factory.

    `generation_config: Dict[str, float]`
    :   Text generation config.
        
        Returns:
            Dict[str, float]: Text generation config.

    `has_tools: bool`
    :   Whether any tools exist in the tool selector.
        
        Returns:
            bool: Whether any tools exist in the tool selector.

    `llm: llmflex.Models.Cores.base_core.BaseLLM`
    :   LLM.
        
        Returns:
            BaseLLM: LLM.

    `memory: llmflex.Memory.long_short_memory.LongShortTermChatMemory`
    :   Current chat memory.
        
        Returns:
            LongShortTermChatMemory: Current chat memory.

    `memory_config: Dict[str, float]`
    :

    `prompt_template: llmflex.Prompts.prompt_template.PromptTemplate`
    :   Prompt template.
        
        Returns:
            PromptTemplate: Prompt template.

    `ranker: llmflex.Rankers.base_ranker.BaseRanker`
    :   Reranker.
        
        Returns:
            BaseRanker: Reranker.

    `text_splitter: llmflex.TextSplitters.base_text_splitter.BaseTextSplitter`
    :   Text splitter.
        
        Returns:
            BaseTextSplitter: Text splitter.

    `tool_selector: llmflex.Tools.tool_utils.ToolSelector`
    :   Tool selector.
        
        Returns:
            ToolSelector: Tool selector.

    `tool_status: Dict[str, bool]`
    :   Whether each tool is on or off.
        
        Returns:
            Dict[str, bool]: Whether each tool is on or off.

    ### Methods

    `create_memory(self) ‑> None`
    :   Create a new chat memory.

    `drop_memory(self, chat_id: str) ‑> None`
    :   Delete the chat memory give the chat ID.
        
        Args:
            chat_id (str): Chat ID.

    `set_generation_config(self, temperature: Optional[float] = None, max_new_tokens: Optional[int] = None, top_p: Optional[float] = None, top_k: Optional[int] = None, repetition_penalty: Optional[float] = None) ‑> None`
    :   Update the LLM generation config. If None is given to any arguments, the argument will not change.
        
        Args:
            temperature (Optional[float], optional): Set how "creative" the model is, the smaller it is, the more static of the output. Defaults to None.
            max_new_tokens (Optional[int], optional): Maximum number of tokens to generate by the llm. Defaults to None.
            top_p (Optional[float], optional): While sampling the next token, only consider the tokens above this p value. Defaults to None.
            top_k (Optional[int], optional): While sampling the next token, only consider the top "top_k" tokens. Defaults to None.
            repetition_penalty (Optional[float], optional): The value to penalise the model for generating repetitive text. Defaults to None.

    `set_memory_config(self, recent_token_limit: Optional[int] = None, relevant_token_limit: Optional[int] = None, relevance_score_threshold: Optional[float] = None, similarity_score_threshold: Optional[float] = None) ‑> None`
    :   Update the memory config. If None is given to any arguments, the argument will not change.
        
        Args:
        recent_token_limit (Optional[int], optional): Token limit for the most recent conversation history. Defaults to None.
        relevant_token_limit (Optional[int], optional): Token limit for the relevant contents from older conversation history. Defaults to None.
        relevance_score_threshold (Optional[float], optional): Score threshold for the reranker for relevant conversation history content extraction. Defaults to None.
        similarity_score_threshold (Optional[float], optional): Score threshold for the vector database search for relevant conversation history content extraction. Defaults to None.

    `set_prompt_template(self, preset: str) ‑> None`
    :   Updating prompt template.
        
        Args:
            preset (str): Preset name of the prompt template.

    `set_system_message(self, system: str) ‑> None`
    :   Update the system message of the current conversation.
        
        Args:
            system (str): Update the system message of the current conversation.

    `switch_memory(self, chat_id: str) ‑> None`
    :   Switch to the memory given the chat ID.
        
        Args:
            chat_id (str): Chat ID.

    `toggle_tool(self, tool_name: str) ‑> None`
    :   Toggle the on/off status of the given tool.
        
        Args:
            tool_name (str): Tool to toggle.