Module llmflex.TextSplitters.sentence_token_text_splitter
=========================================================

Classes
-------

`SentenceTokenTextSplitter(count_token_fn: Callable[[str], int], language_model: str = 'en_core_web_sm', chunk_size: int = 400, chunk_overlap: int = 40)`
:   Text splitter that split text by sentences and group by token counts.
        
    
    Initialise the TextSplitter.
    
    Args:
        count_token_fn (Callable[[str], int]): Function to count the number of tokens in a string.
        language_model (str, optional): Name of the SpaCy model to use. Defaults to 'en_core_web_sm'.
        chunk_size (int, optional): Maximum number of tokens per text chunk. Defaults to 400.
        chunk_overlap (int, optional): Numbers of tokens that overlaps for each subsequent chunks. Defaults to 40.

    ### Ancestors (in MRO)

    * langchain.text_splitter.TextSplitter
    * langchain_core.documents.transformers.BaseDocumentTransformer
    * abc.ABC

    ### Methods

    `split_text(self, text: str) ‑> List[str]`
    :   Splitting the given text.
        
        Args:
            text (str): Text to split.
        
        Returns:
            List[str]: List of split texts.