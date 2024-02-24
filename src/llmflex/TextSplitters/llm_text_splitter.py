from typing import List
from langchain.text_splitter import TextSplitter
from ..Models.Cores.base_core import BaseLLM
from ..Models.Factory.llm_factory import LlmFactory
from typing import Union, Type

class LLMTextSplitter(TextSplitter):
    """Text splitter using the tokenizer in the llm as a measure to count tokens and split texts.
    """

    def __init__(self, model: Union[LlmFactory, Type[BaseLLM]],
                 chunk_size: int = 400, chunk_overlap: int = 40) -> None:
        """Initialise the TextSplitter.

        Args:
            model (Union[LlmFactory, Type[BaseLLM]]): Llm factory that contains the model or the llm that will be used to count tokens.
            chunk_size (int, optional): Maximum number of tokens per text chunk. Defaults to 400.
            chunk_overlap (int, optional): Numbers of tokens that overlaps for each subsequent chunks. Defaults to 40.
        """
        
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._model = model if hasattr(model, 'generation_config') else model()

    def split_text(self, text: str) -> List[str]:
        """Splitting the given text.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of split texts.
        """
        from math import ceil
        token_ids = self._model.get_token_ids(text)
        size = len(token_ids)
        batch_size = self._chunk_size - self._chunk_overlap
        batches = list(map(lambda x: (x * batch_size, min((x + 1) * batch_size + self._chunk_overlap, size)), range(ceil(size / batch_size))))
        batches = list(map(lambda x: token_ids[x[0]:x[1]], batches))
        chunks = list(map(lambda x: self._model.core.decode(x), batches))
        return chunks