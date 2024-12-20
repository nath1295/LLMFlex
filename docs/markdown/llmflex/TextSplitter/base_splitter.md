Module llmflex.TextSplitter.base_splitter
=========================================

Classes
-------

`BaseTextSplitter()`
:   Base text splitter class.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * llmflex.TextSplitter.markdown_splitter.MarkdownTextSplitter
    * llmflex.TextSplitter.recursive_splitter.RecursiveTextSplitter
    * llmflex.TextSplitter.sentence_splitter.SentenceTextSplitter
    * llmflex.TextSplitter.token_splitter.TokenCountTextSplitter

    ### Methods

    `split_documents(self, docs: List[llmflex.Schema.document.Document]) ‑> List[llmflex.Schema.document.Document]`
    :   Splits the given list of documents into a list of documents, preserving the original metadata.
        
        This method iterates over the input list of documents, splits the text of each document using the
        `split_text` method, and creates new documents with the split text while preserving the original metadata.
        
        Args:
            docs (List[Document]): The list of documents to split.
        
        Returns:
            List[Document]: A list of documents resulting from the split, preserving the original metadata.

    `split_text(self, text: str) ‑> List[str]`
    :   Splits the given text into a list of strings.
        
        Args:
            text (str): The text to split.
        
        Returns:
            List[str]: A list of strings resulting from the split.