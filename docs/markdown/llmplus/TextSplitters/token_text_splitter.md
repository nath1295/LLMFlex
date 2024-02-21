Module llmplus.TextSplitters.token_text_splitter
================================================

Classes
-------

`TokenCountTextSplitter(encode_fn: Callable[[str], List[int]], decode_fn: Callable[[List[int]], str], chunk_size: int = 400, chunk_overlap: int = 40)`
:   Text splitter that count tokens and split texts.
        
    
    Initialise the TextSplitter.
    Args:
        encode_fn (Callable[[str], List[int]]): Function of encode a string.
        decode_fn (Callable[[List[int]], str]): Function to decode a list of token ids.
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