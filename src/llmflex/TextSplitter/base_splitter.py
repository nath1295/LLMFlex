from abc import ABC, abstractmethod
from ..Schema.document import Document
from typing import List

class BaseTextSplitter(ABC):
    """Base text splitter class.
    """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into a list of strings.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of strings resulting from the split.
        """
        pass

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Splits the given list of documents into a list of documents, preserving the original metadata.

        This method iterates over the input list of documents, splits the text of each document using the
        `split_text` method, and creates new documents with the split text while preserving the original metadata.

        Args:
            docs (List[Document]): The list of documents to split.

        Returns:
            List[Document]: A list of documents resulting from the split, preserving the original metadata.
        """
        def split_doc(doc: Document) -> List[Document]:
            texts = self.split_text(text=doc.text)
            metadata = doc.metadata
            return [Document(text=text, metadata=metadata.copy()) for text in texts] # Create a copy to avoid modifying the original metadata
        split_docs = [split_doc(doc) for doc in docs]
        return sum(split_docs, [])
