from ..Schemas.documents import Document
from typing import List
from abc import ABC, abstractmethod

class BaseTextSplitter(ABC):
    """Base class for text splitter.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 40) -> None:
        """Initialise the TextSplitter.

        Args:
            chunk_size (int, optional): Maximum number of tokens per text chunk. Defaults to 400.
            chunk_overlap (int, optional): Numbers of tokens that overlaps for each subsequent chunks. Defaults to 40.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Splitting the given text.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of split texts.
        """
        pass

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split the list of given documents.

        Args:
            docs (List[Document]): Documents to split.

        Returns:
            List[Document]: List of splitted documents.
        """
        def split_doc(doc: Document):
            text = doc.index
            metadata = doc.metadata
            text_ls = self.split_text(text)
            new_docs = list(map(lambda x: Document(index=x[0], metadata=x[1]), list(zip(text_ls, [metadata] * len(text_ls)))))
            return new_docs
        new_docs = list(map(split_doc, docs))
        return sum(new_docs, [])