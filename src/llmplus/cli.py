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
@click.option('--embeddings', default='thenlper/gte-small', help='Embeddings model ID to use.')
@click.option('--web_search', is_flag=True, help='Whether to use web search in the interface or not.')
@click.option('--model_type', default='auto', help='LLM model type.')
@click.option('--auth', type=(str, str), default=None, help='User name and password for authentication.')
@click.option('--appname', default='LLMPlus', help='Name of the webapp.')
@click.option('--extras', default='', help='Extra arugments for loading the model.')
def interface(model_id: str = 'TheBloke/OpenHermes-2.5-Mistral-7B-GGUF', 
              embeddings: str = 'thenlper/gte-small', 
              web_search: bool = False,
              model_type: str = 'auto',
              auth: Optional[Tuple[str, str]] = None, 
              appname: str = 'LLMPlus',
              extras: str = "") -> None:
    """Launch the Streamlit Chat GUI.
    """
    from .Frontend.streamlit_interface import run_streamlit_interface
    model_id = None if model_id == 'None' else model_id
    model = dict(model_id=model_id, model_type=model_type, **args_from_string(extras))
    embeddings = dict(embeddings_class='HuggingfaceEmbeddingsToolkit', model_id=embeddings)
    tools = [dict(tool_class='WebSearchTool', embeddings=True, verbose=False)] if web_search else []
    app = run_streamlit_interface(model_kwargs=model, embeddings_kwargs=embeddings, tool_kwargs=tools, auth=auth, debug=False, app_name=appname)

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
    from .utils import get_config
    import os

    model_dir = get_model_dir(model_id=model_id, model_file=model_file)
    kobold_dir = get_config()['llmplus_home'] if kobold_dir is None else kobold_dir
    kobold_dir = os.path.join(kobold_dir, 'koboldcpp', 'koboldcpp.py')
    if not os.path.exists(kobold_dir):
        print(f'Cannot find the script "{kobold_dir}". Falling back to use llama-cpp-python for serving.')
        os.system(f'python -m llama_cpp.server --model {model_dir} --n_ctx {context_size} --use_mlock True --port {port} {extras}')
    else:
        os.system(f'python {kobold_dir} {model_dir} --smartcontext --contextsize {context_size} --port {port} {extras}')
    
    

if __name__=='__main__':
    cli()