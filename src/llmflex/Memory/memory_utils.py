from .base_memory import BaseChatMemory
from ..Models.Cores.base_core import BaseLLM
from ..KnowledgeBase.knowledge_base import KnowledgeBase
from ..Prompts import PromptTemplate
from ..Tools.tool_utils import ToolSelector
from typing import Optional, Type, Dict, List, Union

def create_prompt_with_history(memory: Type[BaseChatMemory], user_input: str, 
        llm: Type[BaseLLM], prompt_template: Optional[PromptTemplate] = None,
        system: Optional[str] = None, recent_token_limit: int = 400, relevant_token_limit: int = 300, relevance_score_threshold: float = 0.8, 
        similarity_score_threshold: float = 0.5, tool_selector: Optional[ToolSelector] = None, 
        knowledge_base: Optional[KnowledgeBase] = None, kb_token_limit: int = 500, kb_score_threshold: float = 0.0) -> Union[str, List[Dict[str, str]]]:
    """Create the full prompt or a list of messages that can be passed to the prompt template given the conversation memory.

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
    """
    from .long_short_memory import LongShortTermChatMemory
    import json
    prompt_template = llm.core.prompt_template if prompt_template is None else prompt_template
    system = memory.system if system is None else system
    user_input = user_input.strip()
    if isinstance(memory, LongShortTermChatMemory):
        mem_type = 'longshort'
    else:
        mem_type = 'base'
    
    messages = [dict(role='system', content=system.strip())]
    return_list: bool = False
    # Adding function metadata if tools are available
    if tool_selector is not None:
        if not tool_selector.is_empty:
            if prompt_template.allow_custom_role:
                messages.append(tool_selector.function_metadata)
                return_list = True
    short_mem = memory.get_token_memory(llm=llm, token_limit=recent_token_limit)
    messages.extend(prompt_template.format_history(history=short_mem, return_list=True))
    extra = dict()
    if knowledge_base:
        kb_content = knowledge_base.search(query=user_input, token_limit=kb_token_limit, relevance_score_threshold=kb_score_threshold)
        if kb_content:
            if prompt_template.allow_custom_role:
                kb_content = list(map(lambda x: dict(content=x.index, source=x.metadata['filename']), kb_content))
                messages.append(dict(role='relevant_contents_from_knowledge_base', content=json.dumps(kb_content, indent=4)))
            else:
                extra['relevant_contents_from_knowledge_base'] = list(map(lambda x: dict(content=x.index, source=x.metadata['filename']), kb_content))
    if mem_type =='longshort':
        long_mem = memory.get_long_term_memory(query=user_input, recent_history=short_mem, llm=llm, 
                token_limit=relevant_token_limit, similarity_score_threshold=similarity_score_threshold, relevance_score_threshold=relevance_score_threshold)
        if long_mem:
            if prompt_template.allow_custom_role:
                messages.append(
                    dict(role='relevant_contents_from_previous_conversation', content=json.dumps(long_mem, indent=4))
                )
            else:
                extra['relevant_contents_from_previous_conversation'] = long_mem
    content = json.dumps(extra, indent=4) + '\n\n' + user_input if len(list(extra.keys())) != 0 else user_input
    messages.append(dict(role='user', content=content))
    if return_list:
        return messages
    else:
        return prompt_template.create_custom_prompt(messages=messages)
