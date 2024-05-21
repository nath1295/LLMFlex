from .base_text_splitter import BaseTextSplitter
from spacy.language import Language
from typing import Callable, List, Union

class SentenceTokenTextSplitter(BaseTextSplitter):
    """Text splitter that split text by sentences and group by token counts.
    """
    def __init__(self, count_token_fn: Callable[[str], int], language_model: Union[str, Language] = 'en_core_web_sm', chunk_size: int = 400, chunk_overlap: int = 40) -> None:
        """Initialise the TextSplitter.

        Args:
            count_token_fn (Callable[[str], int]): Function to count the number of tokens in a string.
            language_model (Union[str, Language], optional): Name of the SpaCy model or the SpaCy model to use. Defaults to 'en_core_web_sm'.
            chunk_size (int, optional): Maximum number of tokens per text chunk. Defaults to 400.
            chunk_overlap (int, optional): Numbers of tokens that overlaps for each subsequent chunks. Defaults to 40.
        """
        import spacy
        from spacy.cli import download
        from spacy.util import is_package
        if not is_package(language_model):
            try:
                download(language_model)
            except:
                raise RuntimeError(f'Failed to download the SpaCy languange model "{language_model}".')
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.nlp = spacy.load(language_model) if isinstance(language_model, str) else language_model
        self.count_fn = count_token_fn
        self._language_model = language_model
    
    def split_text(self, text: str) -> List[str]:
        """Splitting the given text.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of split texts.
        """
        doc = self.nlp(text)
        sentences = list(map(lambda x: x.text, doc.sents))
        sent_counts = list(map(self.count_fn, sentences))
        chunks = []
        current_chunk = []
        current_count = 0
        last_count = 0
        for i, sent in enumerate(sentences):
            sent_ct = sent_counts[i]
            if (sent_ct + current_count) <= self._chunk_size:
                current_count += sent_ct
                current_chunk.append(sent)
                last_count = sent_ct
            else:
                if len(current_chunk) != 0:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i - 1]] if ((last_count <= self._chunk_overlap) and (i != 0)) else []
                current_count = sent_counts[i - 1] if ((last_count <= self._chunk_overlap) and (i != 0)) else 0
                current_count += sent_ct
                current_chunk.append(sent)
                last_count = sent_ct
        if len(current_chunk) != 0:
            chunks.append(' '.join(current_chunk))
        return chunks
