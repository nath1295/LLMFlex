from .base_splitter import BaseTextSplitter
from typing import TYPE_CHECKING, List, Optional, Callable
if TYPE_CHECKING:
    from ..Embeddings.Model.base_embeddings import BaseEmbeddings
    from ..Tokenizer.base_tokenizer import BaseTokenizer
    import numpy as np

class SemanticTextSplitter(BaseTextSplitter):
    """A text splitter that splits text into chunks based on semantic similarity.
    
    This splitter uses embeddings to determine the semantic similarity between chunks of text.
    It splits the text into chunks such that each chunk is as semantically similar as possible to the others.
    """
    def __init__(self, 
                 embeddings: "BaseEmbeddings", 
                 tokenizer: Optional["BaseTokenizer"] = None, 
                 chunk_size: int = 400, 
                 split_fn: Optional[Callable[[str], List[str]]] = None
            ) -> None:
        """Initializes the SemanticTextSplitter.

        Args:
            embeddings (BaseEmbeddings): The embeddings model to use for semantic similarity.
            tokenizer (Optional[BaseTokenizer], optional): The tokenizer to use for counting tokens. If None is given, the tokenizer of the embedding model will be used. Defaults to None.
            chunk_size (int, optional): The maximum size of each chunk. Defaults to 400.
            split_fn (Optional[Callable[[str], List[str]]], optional): A custom function to split the text, If None is given, a sentence text splitter will be used with chunk size being one third of the SemanticTextSplitter chunk size. Defaults to None.
        """
        super().__init__()
        self._embeddings = embeddings
        self._tokenizer = tokenizer if tokenizer else self._embeddings.tokenizer
        self._chunk_size = chunk_size
        self._split_fn = split_fn
        if chunk_size < 40:
            raise ValueError('"chunk_size" must be larger than 40.')
        if not self._split_fn:
            from .sentence_splitter import SentenceTextSplitter
            st_chunk_size = chunk_size // 3
            self._split_fn = SentenceTextSplitter(tokenizer=self._tokenizer, chunk_size=st_chunk_size, chunk_overlap=0).split_text

    def _split_semantic_chunks(self, sentences: List[str], sizes: List[int], vectors: "np.ndarray") -> List[str]:
        """Splits the given sentences into chunks based on semantic similarity.
        
        Args:
            sentences (List[str]): The list of sentences to split.
            sizes (List[int]): The sizes of each sentence.
            vectors (np.ndarray): The embeddings of each sentence.

        Returns:
            List[str]: The list of split sentences.
        """
        import numpy as np
        scores = (vectors[:-1] * vectors[1:]).sum(axis=1)
        min_score = scores.min()
        bps = np.where(scores<=min_score)[0] + 1
        batches = [(bps[i - 1] if i != 0 else 0, bp) for i, bp in enumerate(bps)] + [(bps[-1], len(sentences))]
        chunks = [sentences[i:j] for i, j in batches]
        chunk_size_arr = [sizes[i:j] for i, j in batches]
        chunk_sizes = [sum(csa) for csa in chunk_size_arr]
        chunk_vectors = [vectors[i:j] for i, j in batches]
        outputs = []
        for chunk, size_arr, total_size, vecs in zip(chunks, chunk_size_arr, chunk_sizes, chunk_vectors):
            if total_size <= self._chunk_size:
                outputs.append(''.join(chunk))
            elif len(chunk) <= 1:
                outputs.extend(chunk)
            else:
                outputs.extend(self._split_semantic_chunks(chunk, size_arr, vecs))
        return outputs


    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into a list of strings.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of strings resulting from the split.
        """
        sentences = self._split_fn(text)
        sizes = [self._tokenizer.get_num_tokens(c, add_special_tokens=False) for c in sentences]
        if all(size >= self._chunk_size for size in sizes):
            return sentences
        elif len(sentences) <= 1:
            return sentences
        vectors = self._embeddings.batch_embed(sentences)
        return self._split_semantic_chunks(sentences, sizes, vectors)
        
