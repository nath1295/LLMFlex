import os

class HuggingFaceEmbeddingsServer:

    def __init__(self, model_id: str = 'thenlper/gte-small', default_batch_size: int = 128, **kwargs) -> None:
        """Initialising the model server.

        Args:
            model_id (str, optional): Huggingface repo id. Defaults to 'thenlper/gte-small'.
            default_batch_size (int, optional): Default batch size for encoding if not specified on the client side. Defaults to 128.
        """
        from flask import Flask
        from ..utils import get_config
        from sentence_transformers import SentenceTransformer
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = get_config()['st_home']
        os.environ['HF_HOME'] = get_config()['hf_home']
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.model = SentenceTransformer(model_id, **kwargs)
        self.default_batch_size = default_batch_size
        self.info = dict(
            model_id=model_id,
            embedding_dimension=self.model.get_sentence_embedding_dimension(),
            max_seq_length=self.model.max_seq_length,
            device=str(self.model.device),
            default_batch_size=default_batch_size
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
            batch_size =  args_dict.get('batch_size', self.default_batch_size)
            normalize_embddings = args_dict.get('normalize_embeddings', True)
            embeddings = self.model.encode(input_texts, batch_size=batch_size, normalize_embeddings=normalize_embddings).tolist()
            if self.model.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.model.device.type == 'cuda':
                torch.cuda.empty_cache()
            else:
                import gc
                gc.collect()
            return jsonify(embeddings)
        
        @self.app.route('/info', methods=['GET'])
        def get_info():
            return jsonify(self.info)
        
        self.app.run(**kwargs)
