import click
from .utils import PACKAGE_DISPLAY_NAME
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
    """Setting paths for the package.
    """
    from .utils import set_config, get_config
    config = get_config()
    new_config = dict()
    print('Setting paths:')
    for k, v in config.items():
        new = input(f'{k} [{v}]: ')
        new_config[k] = new.strip() if new.strip() != '' else v
    
    set_config(**new_config)

@cli.command()
@click.option('--config_dir', default=None, help='Config file to load the webapp.')
def interface(config_dir: str) -> None:
    """Launch the Streamlit Chat GUI.
    """
    from .Frontend.streamlit_interface import DEFAULT_CONFIG, create_streamlit_script
    import os
    import yaml
    script_dir = create_streamlit_script()
    if config_dir is None:
        config = DEFAULT_CONFIG
    else:
        with open(config_dir, 'r') as f:
            config = yaml.safe_load(f)
    with open(os.path.join(os.path.dirname(script_dir), 'chatbot_config.yaml'), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    os.system(f'streamlit run {script_dir}')

@cli.command()
@click.option('--filename', default='chatbot_config.yaml', help='LLM model ID to use.')
def create_app_config(filename: str = 'chatbot_config.yaml') -> None:
    """Generate a chatbot config file template.
    """
    from .Frontend.streamlit_interface import DEFAULT_CONFIG
    import yaml
    with open(filename, 'w') as f:
        yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)

@cli.command()
@click.option('--model_id', default='TheBloke/OpenHermes-2.5-Mistral-7B-GGUF', help='LLM model ID to use.')
@click.option('--model_file', default=None, help='Model quant file to use. Defaults to None.')
@click.option('--context_size', default=4096, help='Context size of the model. Defaults to 4096.')
@click.option('--port', default=5001, help='Port to use. Defaults to 5001.')
@click.option('--kobold_dir', default=None, help='Directory of the KoboldCPP. Defaults to "koboldcpp" under home directory.')
@click.option('--extras', default='', help='Extra arugments for KoblodCPP or llama-cpp-python.')
def serve(model_id: str, model_file: Optional[str] = None, context_size: int = 4096, port: int = 5001, kobold_dir: str = '', extras: str = '') -> None:
    """Serve a llm with GGUF format from HuggingFace.
    """
    from .Models.Cores.llamacpp_core import get_model_dir
    import os

    model_dir = get_model_dir(model_id=model_id, model_file=model_file)
    os.system(f'python -m llama_cpp.server --model {model_dir} --n_ctx {context_size} --use_mlock True --port {port} {extras}')
    
    

if __name__=='__main__':
    cli()