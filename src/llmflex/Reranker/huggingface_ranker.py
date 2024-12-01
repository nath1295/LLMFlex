from .base_ranker import RankResult, BaseRanker
from ..Schema.document import Document
from typing import List, Union, Dict, Any, Optional

class HuggingFaceRanker(BaseRanker):
    """A ranker that uses the HuggingFace transformers library for ranking documents.

    This ranker uses a pre-trained transformer model from the HugginFace transformers library to score and rank documents.
    """
    def __init__(self, pretrained_model_name_or_path: str, model_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initializes the HuggingFaceRanker with a pretrained model and optional model and tokenizer kwargs.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
            model_kwargs (Optional[Dict[str, Any]], optional): Optional keyword arguments to pass to the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Optional keyword arguments to pass to the tokenizer. Defaults to None.
        """
        from ..utils import get_config
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_kwargs = dict() if model_kwargs is None else model_kwargs
        tokenizer_kwargs = dict() if tokenizer_kwargs is None else tokenizer_kwargs
        model_cache_dir = model_kwargs.pop('cache_dir', get_config('hf_home'))
        tokenizer_cache_dir = tokenizer_kwargs.pop('cache_dir', get_config('hf_home'))
        self._model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=model_cache_dir, **model_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=tokenizer_cache_dir, **tokenizer_kwargs)
        super().__init__(model_id=pretrained_model_name_or_path)

    def rerank(self, query: str, docs: List[Union[str, Document]], top_k: Optional[int] = None) -> List[RankResult]:
        """Reranks the given list of documents based on the provided query using the model associated with this ranker.

        Args:
            query (str): The query string to use for reranking.
            docs (List[Union[str, Document]]): The list of documents to rerank. Each document can be either a string or a Document object.
            top_k (Optional[int], optional): The maximum number of documents to return. If not provided, all documents will be returned. Defaults to None.
            
        Returns:
            List[RankResult]: A list of RankResult tuples, where each tuple contains a Document and its corresponding score.
        """
        if (top_k is not None) and (top_k < 1):
            raise ValueError('"top_k" must be an integer larger than 0.')
        import torch
        num_docs = len(docs)
        top_k = num_docs if top_k is None else min(top_k, num_docs)
        docs = [Document(text=doc) if isinstance(doc, str) else doc for doc in docs]
        strings = [doc.text for doc in docs]
        features = self._tokenizer(
                [query] * num_docs, strings,  padding=True, truncation=True, return_tensors="pt").to(self._model.device)
        self._model.eval()
        with torch.no_grad():
            scores = self._model(**features).logits.squeeze(1).detach().numpy()
        if self._model.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self._model.device.type == 'mps':
            torch.mps.empty_cache()

        results = [RankResult(doc=doc, score=score) for doc, score in zip(docs, scores)]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

