from typing import Dict, Any, Optional

class HFEmbeddingServer:
    """Class for serving a HuggingFace embedding model on an API server.
    """
    def __init__(self, pretrained_model_name_or_path: str, 
        batch_size: int = 256, normalize: bool = True, model_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initialising the server class.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
            batch_size (int, optional): The batch size to use for tokenization and embedding generation. Defaults to 256.
            normalize (bool, optional): Whether to normalize the embeddings. Defaults to True.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the Hugging Face model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the Hugging Face tokenizer. Defaults to None.
        """
        try:
            from flask import Flask
        except:
            raise ModuleNotFoundError(f'"flask" not installed. Install with `pip install flask`.')
        from .Model.huggingface_embeddings import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            batch_size=batch_size,
            verbose=False,
            normalize=normalize,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs
        )
        self.info = dict(
            model_id=self.embeddings.model_id,
            max_seq_len=self.embeddings.max_seq_len,
            embedding_size=self.embeddings.embedding_size
        )
        self.app = Flask(__name__)

    def run(self, **kwargs) -> None:
        """Start the server.
        """
        from flask import request, jsonify
        import torch
        @self.app.route('/embeddings', methods=['GET'])
        def get_embeddings():
            args_dict = request.json
            input_texts = args_dict.get('input_texts')
            batch_size =  args_dict.get('batch_size', self.embeddings._batch_size)
            normalize = args_dict.get('normalize', self.embeddings._normalize)
            model_normalize = self.embeddings._normalize
            model_batch_size = self.embeddings._batch_size
            self.embeddings._batch_size = batch_size
            self.embeddings._normalize = normalize
            embeddings = self.embeddings.batch_embed(input_texts).tolist()
            self.embeddings._batch_size = model_batch_size
            self.embeddings._normalize = model_normalize
            if self.embeddings._model.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.embeddings._model.device.type == 'cuda':
                torch.cuda.empty_cache()
            else:
                import gc
                gc.collect()
            return jsonify(embeddings)
        
        @self.app.route('/info', methods=['GET'])
        def get_info():
            return jsonify(self.info)
        
        self.app.run(**kwargs)