Module llmflex.TextSplitters.llm_text_splitter
==============================================

Classes
-------

`LLMTextSplitter(model: Union[llmflex.Models.Factory.llm_factory.LlmFactory, Type[llmflex.Models.Cores.base_core.BaseLLM]], chunk_size: int = 400, chunk_overlap: int = 40)`
:   Text splitter using the tokenizer in the llm as a measure to count tokens and split texts.
        
    
    Initialise the TextSplitter.
    
    Args:
        model (Union[LlmFactory, Type[BaseLLM]]): Llm factory that contains the model or the llm that will be used to count tokens.
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