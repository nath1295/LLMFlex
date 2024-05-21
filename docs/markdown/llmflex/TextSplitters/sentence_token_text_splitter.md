Module llmflex.TextSplitters.sentence_token_text_splitter
=========================================================

Classes
-------

`SentenceTokenTextSplitter(count_token_fn: Callable[[str], int], language_model: Union[str, spacy.language.Language] = 'en_core_web_sm', chunk_size: int = 400, chunk_overlap: int = 40)`
:   Text splitter that split text by sentences and group by token counts.
        
    
    Initialise the TextSplitter.
    
    Args:
        count_token_fn (Callable[[str], int]): Function to count the number of tokens in a string.
        language_model (Union[str, Language], optional): Name of the SpaCy model or the SpaCy model to use. Defaults to 'en_core_web_sm'.
        chunk_size (int, optional): Maximum number of tokens per text chunk. Defaults to 400.
        chunk_overlap (int, optional): Numbers of tokens that overlaps for each subsequent chunks. Defaults to 40.

    ### Ancestors (in MRO)

    * llmflex.TextSplitters.base_text_splitter.BaseTextSplitter
    * abc.ABC