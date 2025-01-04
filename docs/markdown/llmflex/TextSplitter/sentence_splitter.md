Module llmflex.TextSplitter.sentence_splitter
=============================================

Functions
---------

`split_sentences(text: str) ‑> List[str]`
:   

Classes
-------

`SentenceTextSplitter(tokenizer: BaseTokenizer, chunk_size: int = 400, chunk_overlap: int = 40)`
:   Text splitter that split texts into sentences.
        
    
    Initialize the TextSplitter.
    
    Args:
        tokenizer (BaseTokenizer): An instance of the BaseTokenizer class.
        chunk_size (int, optional): The maximum number of tokens per text chunk. Defaults to 400.
        chunk_overlap (int, optional): The number of tokens that overlaps for each subsequent chunk. Defaults to 40.

    ### Ancestors (in MRO)

    * llmflex.TextSplitter.base_splitter.BaseTextSplitter
    * abc.ABC