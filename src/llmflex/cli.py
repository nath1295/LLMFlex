

def main():
    import argparse
    from typing import Optional
    from .utils import PACKAGE_NAME, set_config, get_config

    parser = argparse.ArgumentParser(prog=PACKAGE_NAME, description=f'Welcome to {PACKAGE_NAME} CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Subcommand for setting default config
    def set_config_cli() -> None:
        config = get_config('all')
        new_config = dict()
        for k, v in config.items():
            u = input(prompt=f'{k} [{v}]: ')
            u = v if u.strip() == '' else u
            new_config[k] = u
        set_config(**new_config)

    parser_set_config = subparsers.add_parser('config', help='Set default directories. "package_home" is the default working directory, "hf_home" is the default HuggingFace cache directory.')
    parser_set_config.set_defaults(func=set_config_cli)

    # Subcommand for serving embedding model
    def serve_embeddings(
            model_id: str,
            batch_size: int = 256,
            normalize: bool = True,
            model_kwargs: Optional[str] = None,
            tokenizer_kwargs: Optional[str] = None,
            port: int = 5003,
            host: Optional[str] = None
    ) -> None:
        from .Embeddings.hf_embedding_server import HFEmbeddingServer
        import json
        app = HFEmbeddingServer(
            pretrained_model_name_or_path=model_id,
            batch_size=batch_size,
            normalize=normalize,
            model_kwargs=model_kwargs if model_kwargs is None else json.loads(model_kwargs),
            tokenizer_kwargs=tokenizer_kwargs if tokenizer_kwargs is None else json.loads(tokenizer_kwargs)
        )
        run_kwargs = dict(port=port)
        if host:
            run_kwargs['host'] = host
        app.run(**run_kwargs)

    parser_embd = subparsers.add_parser('serve-embedding', help='Serve an embedding model.')
    parser_embd.add_argument('--model-id', type=str, help='HuggingFace repository id or local path of the embedding model.')
    parser_embd.add_argument('--batch-size', type=int, default=256, help='Default batch size of creating embeddings from strings.')
    parser_embd.add_argument('--normalize', type=bool, default=False, help='Default whether embedded vectors should be normalized. Defaults to True.')
    parser_embd.add_argument('--model-kwargs', type=str, default=None, help='Arguments for loading the model in a json string.')
    parser_embd.add_argument('--tokenizer-kwargs', type=str, default=None, help='Arguments for loading the tokenizer in a json string.')
    parser_embd.add_argument('--port', type=int, default=5003, help='Port of the embedding server. Defaults to 5003.')
    parser_embd.add_argument('--host', type=str, default=None, help='Host of the server. Defaults to None.')
    parser_embd.set_defaults(func=serve_embeddings)

    args = parser.parse_args()
    if args.command:
        args_kwargs = vars(args)
        args.func(**{k: v for k, v in args_kwargs.items() if k not in ['command', 'func']})
    else:
        parser.print_help()

__doc__ = """
This script provides a set of commands to manage the package and serve embedding models.

Commands:
- `config`: Set default directories for the package.
- `serve-embedding`: Serve an embedding model using a HuggingFace model.

For more information on each command, use the `--help` option with the command name.
"""

if __name__ == '__main__':
    main()