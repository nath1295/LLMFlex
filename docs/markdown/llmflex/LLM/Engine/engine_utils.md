Module llmflex.LLM.Engine.engine_utils
======================================

Functions
---------

`convert_tokenizer(tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer) ‑> outlines.models.tokenizer.Tokenizer`
:   

`get_choice_processor(choices: List[str], tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer) ‑> outlines.processors.structured.RegexLogitsProcessor`
:   

`get_grammar_processor(cfg_str: str, tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer) ‑> outlines.processors.structured.CFGLogitsProcessor`
:   

`get_json_processor(schema: Dict[str, Any] | str, tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer, whitespace_pattern: str | None = None) ‑> outlines.processors.structured.JSONLogitsProcessor`
:   

`get_regex_processor(regex_str: str, tokenizer: llmflex.Tokenizer.base_tokenizer.BaseTokenizer) ‑> outlines.processors.structured.RegexLogitsProcessor`
: