import click
from typing import Optional, Tuple

@click.group()
def cli() -> None:
    pass

@cli.command()
def config() -> None:
    """Setting paths for LlmPlus.
    """
    from .utils import set_config, get_config
    config = get_config()
    new_config = dict()
    print('Setting paths for LlmPlus:')
    for k, v in config.items():
        new = input(f'{k} [{v}]: ')
        new_config[k] = new.strip() if new.strip() != '' else v
    
    set_config(**new_config)

@cli.command()
@click.option('--model_id', default='TheBloke/OpenHermes-2.5-Mistral-7B-GGUF', help='LLM model ID to use.')
@click.option('--embeddings', default='thenlper/gte-large', help='Embeddings model ID to use.')
@click.option('--model_type', default='auto', help='LLM model type.')
@click.option('--mobile', is_flag=True, help='Whether to launch the mobile interface or not.')
@click.option('--auth', type=(str, str), default=None, help='User name and password for authentication.')
@click.option('--share', is_flag=True, help='Whether to create a public link or not.')
def interface(model_id: str = 'TheBloke/OpenHermes-2.5-Mistral-7B-GGUF', 
              embeddings: str = 'thenlper/gte-large', 
              model_type: str = 'auto',
              mobile: bool = False, auth: Optional[Tuple[str, str]] = None,
              share: bool = False) -> None:
    """Launch the Gradio Chat GUI.
    """
    from . import HuggingfaceEmbeddingsToolkit, LlmFactory
    from .Frontend.chat_interface import ChatInterface

    model = LlmFactory(model_id=model_id, model_type=model_type)
    embeddings = HuggingfaceEmbeddingsToolkit(model_id=embeddings)
    app = ChatInterface(model=model, embeddings=embeddings)
    app.launch(mobile=mobile, auth=auth, share=share)

if __name__=='__main__':
    cli()