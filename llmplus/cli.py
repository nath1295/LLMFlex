import click
from typing import Optional, Tuple, Dict, Any

def args_from_string(arg_string: str) -> Dict[str, Any]:
    """Parsing kwargs from a string.

    Args:
        arg_string (str): String of arguments.

    Returns:
        Dict[str, Any]: kwargs.
    """
    import ast
    args = {}
    buf = ""
    stack = []
    for ch in arg_string:
        if ch in ['{', '[']:
            stack.append(ch)
        elif ch in ['}', ']']:
            stack.pop()
        elif ch == ',' and not stack:
            key, value = map(str.strip, buf.split('=', 1))
            args[key] = ast.literal_eval(value)
            buf = ""
            continue
        buf += ch

    # Handling the last key-value pair (or the only pair if no comma is present)
    if buf:
        key, value = map(str.strip, buf.split('=', 1))
        args[key] = ast.literal_eval(value)

    return args

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
@click.option('--extra', default='', help='Extra arugments for loading the model.')
def interface(model_id: str = 'TheBloke/OpenHermes-2.5-Mistral-7B-GGUF', 
              embeddings: str = 'thenlper/gte-large', 
              model_type: str = 'auto',
              mobile: bool = False, auth: Optional[Tuple[str, str]] = None,
              share: bool = False,
              extra: str = "") -> None:
    """Launch the Gradio Chat GUI.
    """
    from . import HuggingfaceEmbeddingsToolkit, LlmFactory
    from .Frontend.chat_interface import ChatInterface
    model_id = None if model_id == 'None' else model_id
    model = LlmFactory(model_id=model_id, model_type=model_type, **args_from_string(extra))
    embeddings = HuggingfaceEmbeddingsToolkit(model_id=embeddings)
    app = ChatInterface(model=model, embeddings=embeddings)
    app.launch(mobile=mobile, auth=auth, share=share)

@cli.command()
@click.option('--model_id', default='TheBloke/OpenHermes-2.5-Mistral-7B-GGUF', help='LLM model ID to use.')
@click.option('--model_file', default=None, help='Model quant file to use. Defaults to None.')
@click.option('--context_size', default=4096, help='Context size of the model. Defaults to 4096.')
@click.option('--port', default=5001, help='Port to use. Defaults to 5001.')
@click.option('--kobold_dir', default='', help='Directory of the KoboldCPP. Defaults to the current directory.')
def serve(model_id: str, model_file: Optional[str] = None, context_size: int = 4096, port: int = 5001, kobold_dir: str = '') -> None:
    """Serve a llm with GGUF format from HuggingFace.
    """
    from . import LlmFactory
    from .Models.Factory.llm_factory import detect_model_type
    import os

    model = LlmFactory(model_id=model_id, model_file=model_file, model_type='gguf')
    model_path = model.core.model.model_path
    model.core.unload()
    kobold_dir = os.path.join(kobold_dir, 'koboldcpp.py')
    if not os.path.exists(kobold_dir):
        raise FileNotFoundError(f'Cannot find the script "{kobold_dir}".')
    os.system(f'python {kobold_dir} {model_path} --smartcontext --contextsize {context_size} --port {port}')
    
    

if __name__=='__main__':
    cli()