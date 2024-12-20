from ...Tokenizer.base_tokenizer import BaseTokenizer
from typing import Union, Dict, Any, Optional, List
try:
    from outlines.processors.structured import RegexLogitsProcessor, JSONLogitsProcessor, CFGLogitsProcessor
    from outlines.models.tokenizer import Tokenizer
    outlines_installed = True
except:
    outlines_installed = False

def convert_tokenizer(tokenizer: BaseTokenizer) -> Tokenizer:
    if tokenizer.__class__.__name__ == 'HuggingFaceTokenizer':
        from outlines.models.transformers import TransformerTokenizer
        return TransformerTokenizer(tokenizer=tokenizer.hf_tokenizer)
    elif tokenizer.__class__.__name__ == 'LlamaCppTokenizer':
        from outlines.models.llamacpp import LlamaCppTokenizer
        return LlamaCppTokenizer(model=tokenizer.llama_tokenizer)
    else:
        raise TypeError(f'Tokenizer class "{tokenizer.__class__.__name__}" cannot be converted.')

# Guided decoding logits processors
def get_regex_processor(regex_str: str, tokenizer: BaseTokenizer) -> RegexLogitsProcessor:
    if not outlines_installed:
        raise ModuleNotFoundError('"outlines" not installed. Please install with `pip install outlines`.')
    return RegexLogitsProcessor(regex_str, tokenizer=convert_tokenizer(tokenizer))

def get_json_processor(schema: Union[Dict[str, Any], str], tokenizer: BaseTokenizer, whitespace_pattern: Optional[str] = None) -> JSONLogitsProcessor:
    if not outlines_installed:
        raise ModuleNotFoundError('"outlines" not installed. Please install with `pip install outlines`.')
    return JSONLogitsProcessor(schema=schema, tokenizer=convert_tokenizer(tokenizer), whitespace_pattern=whitespace_pattern)

def get_choice_processor(choices: List[str], tokenizer: BaseTokenizer) -> RegexLogitsProcessor:
    if not outlines_installed:
        raise ModuleNotFoundError('"outlines" not installed. Please install with `pip install outlines`.')
    regex_str = r"(" + r"|".join(choices) + r")"
    return get_regex_processor(regex_str=regex_str, tokenizer=tokenizer)

def get_grammar_processor(cfg_str: str, tokenizer: BaseTokenizer) -> CFGLogitsProcessor:
    if not outlines_installed:
        raise ModuleNotFoundError('"outlines" not installed. Please install with `pip install outlines`.')
    return CFGLogitsProcessor(cfg_str=cfg_str, tokenizer=convert_tokenizer(tokenizer))