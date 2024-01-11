Module llmplus.Models.Cores.text_splitter
=========================================

Classes
-------

`LLMTextSplitter(model: Union[llmplus.Models.Factory.llm_factory.LlmFactory, Type[llmplus.Models.Cores.base_core.BaseLLM]], chunk_size: int = 400, chunk_overlap: int = 40)`
:   Interface for splitting text into chunks.
    
    Create a new TextSplitter.
    
    Args:
        chunk_size: Maximum size of chunks to return
        chunk_overlap: Overlap in characters between chunks
        length_function: Function that measures the length of given chunks
        keep_separator: Whether to keep the separator in the chunks
        add_start_index: If `True`, includes chunk's start index in metadata
        strip_whitespace: If `True`, strips whitespace from the start and end of
                          every document

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