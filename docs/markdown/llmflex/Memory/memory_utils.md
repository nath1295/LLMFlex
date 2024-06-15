Module llmflex.Memory.memory_utils
==================================

Functions
---------

    
`create_prompt_with_history(memory: Type[llmflex.Memory.base_memory.BaseChatMemory], user_input: str, llm: Type[llmflex.Models.Cores.base_core.BaseLLM], prompt_template: Optional[llmflex.Prompts.prompt_template.PromptTemplate] = None, system: Optional[str] = None, recent_token_limit: int = 400, relevant_token_limit: int = 300, relevance_score_threshold: float = 0.8, similarity_score_threshold: float = 0.5, tool_selector: Optional[llmflex.Tools.tool_utils.ToolSelector] = None, knowledge_base: Optional[llmflex.KnowledgeBase.knowledge_base.KnowledgeBase] = None, kb_token_limit: int = 500, kb_score_threshold: float = 0.0) ‑> Union[str, List[Dict[str, str]]]`
:   Create the full prompt or a list of messages that can be passed to the prompt template given the conversation memory.
    
    Args:
        memory (Type[BaseChatMemory]): Conversation memory.
        user_input (str): Latest user query.
        llm (Type[BaseLLM]): LLM for counting token.
        prompt_template (Optional[PromptTemplate], optional): Prompt template to format the prompt. If None is given, the default prompt template of the llm will be used. Defaults to None.
        system (Optional[str], optional): System messsage for the conversation. If None is given, the default system message from the chat memory will be used. Defaults to None.
        recent_token_limit (int, optional): Token limit for the most recent conversation history. Defaults to 400.
        relevant_token_limit (int, optional): Token limit for the relevant contents from older conversation history. Only used if the memory provided allow relevant content extraction or knowledge base is given. Defaults to 300.
        relevance_score_threshold (float, optional): Score threshold for the reranker for relevant conversation history content extraction. Only used if the memory provided allow relevant content extraction or knowledge base is given. Defaults to 0.8.
        similarity_score_threshold (float, optional): Score threshold for the vector database search for relevant conversation history content extraction. Only used if the memory provided allow relevant content extraction. Defaults to 0.5.
        tool_selector (Optional[ToolSelector], optional): Tool selector with all the available tools for function calling. If None is given, function calling is not enabled. Defaults to None.
        knowledge_base (Optional[KnowledgeBase], optional): Knowledge base for RAG. Defaults to None.
        kb_token_limit (int, optional): Knowledge base search token limit. Defaults ot 500.
        kb_score_threshold (float, optional): Knowledge base search relevance score threshold. Defaults to 0.0.
    
    Returns:
        Union[str, List[Dict[str, str]]]: Return the full prompt as a string if function calling is not applicable. Otherwise a list of messages will be returned for further formatting for function calling.