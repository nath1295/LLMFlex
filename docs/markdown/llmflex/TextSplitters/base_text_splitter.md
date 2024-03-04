Module llmflex.TextSplitters.base_text_splitter
===============================================

Classes
-------

`BaseTextSplitter(chunk_size: int = 400, chunk_overlap: int = 40)`
:   Base class for text splitter.
        
    
    Initialise the TextSplitter.
    
    Args:
        chunk_size (int, optional): Maximum number of tokens per text chunk. Defaults to 400.
        chunk_overlap (int, optional): Numbers of tokens that overlaps for each subsequent chunks. Defaults to 40.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.TextSplitters.sentence_token_text_splitter.SentenceTokenTextSplitter
    * llmflex.TextSplitters.token_text_splitter.TokenCountTextSplitter

    ### Methods

    `split_documents(self, docs: List[llmflex.Schemas.documents.Document]) ‑> List[llmflex.Schemas.documents.Document]`
    :   Split the list of given documents.
        
        Args:
            docs (List[Document]): Documents to split.
        
        Returns:
            List[Document]: List of splitted documents.

    `split_text(self, text: str) ‑> List[str]`
    :   Splitting the given text.
        
        Args:
            text (str): Text to split.
        
        Returns:
            List[str]: List of split texts.