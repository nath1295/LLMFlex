import os
import subprocess
import json
from typing import Literal, Union, List, Dict, Any, Optional

### Helper functions
def os_name() -> Literal['Windows', 'Linux', 'MacOS_intel', 'MacOS_apple_silicon', 'MacOS_unknown', 'Unknown']:
    """Get the current operating system.

    Returns:
        Literal['Windows', 'Linux', 'MacOS_intel', 'MacOS_apple_silicon', 'MacOS_unknown', 'Unknown']: The detected OS.
    """
    import platform
    os_name = platform.system()

    if os_name == 'Windows':
        return "Windows"
    elif os_name == 'Linux':
        return "Linux"
    elif os_name == 'Darwin':
        # macOS, check for CPU type
        cpu_arch = os.uname().machine
        if cpu_arch == 'x86_64':
            return "MacOS_intel"
        elif cpu_arch == 'arm64':
            return "MacOS_apple_silicon"
        else:
            return f"MacOS_unknown"
    else:
        return f"Unknown"
    
def is_cuda() -> bool:
    """Whether CUDA is available.

    Returns:
        bool: True if CUDA is available.
    """
    try:
        subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        return False
    
def is_colab() -> bool:
    """Check if it's on Google Colab.
    Returns:
        bool: True if on Google Colab.
    """
    colab = False
    try:
        from google.colab import drive
        colab = True
    except:
        pass
    return colab 

def is_conda() -> bool:
    """Check if it's in a conda environment

    Returns:
        bool: True if in a conda environment.
    """
    return 'CONDA_PREFIX' in os.environ

def env_name() -> str:
    """Get the current python environment name.

    Returns:
        str: Current python environment name.
    """
    if is_colab():
        return 'base'
    import sys
    base = os.path.basename(sys.prefix)
    if base.lower() == 'anaconda3':
        return 'base'
    elif 'python3' in base.lower():
        return 'base'
    else:
        return base
    
def read_json(file_dir: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Read the json file provided as a dictionary or a list of dictionaries.

    Args:
        file_dir (str): Path of the json file.

    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: Content of the json file as a dictionary or a list of dictionaries.
    """
    with open(file_dir, 'r') as f:
        content = json.load(f)
    return content

def save_json(content: Union[Dict[str, Any], List[Dict[str, Any]]], file_dir: str) -> None:
    """Save the given dictionary or list of dictionaries as a json file.

    Args:
        content (Union[Dict[str, Any], List[Dict[str, Any]]]): Dictionary or list of dictionaries to save.
        file_dir (str): Path (with filename) to save the content.
    """
    file_dir = os.path.abspath(file_dir)
    parent = os.path.dirname(file_dir)
    os.makedirs(parent, exist_ok=True)
    with open(file_dir, 'w') as f:
        json.dump(content, f, indent=4)

def current_time() -> float:
    """Getting the current time as a float timestamp, but still human-readable.

    Returns:
        float: Timestamp.
    """
    from datetime import datetime as dt
    now = dt.now().strftime('%Y%m%d%H%M%S.%f')
    return float(now)

### Package configuration

## Defaults
PACKAGE_DISPLAY_NAME = 'LLMFlex'
PACKAGE_NAME = 'llmflex'

colab_home = '/content/drive/MyDrive'

user_home = colab_home if is_colab() else os.path.expanduser('~')

home_dir = os.path.join(colab_home, PACKAGE_DISPLAY_NAME) if is_colab() else os.getcwd()

def get_config_dir() -> str:
    """Get the configuration file path of the package.

    Returns:
        str: configuration file path.
    """
    config_dir = os.path.join(user_home, '.config', PACKAGE_NAME, env_name())
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return os.path.join(config_dir, 'config.json')

def get_config(element: Literal['all', 'package_home', 'hf_home', 'st_home'] = 'all') -> Union[Dict[str, str], str]:
    """Get the configuration of the package.

    Args: 
        element (Literal[&#39;all&#39;, &#39;package_home&#39;, &#39;hf_home&#39;, &#39;st_home&#39;], optional): The output element of the configuration. Defaults to 'all'.

    Returns:
        Union[Dict[str, str], str]: Configuration of the package or one of hte configured directories.
    """
    config_dir = get_config_dir()

    default_config = dict(
        package_home = home_dir,
        hf_home = os.path.join(home_dir, 'hf_home') if is_colab() else os.path.join(user_home, '.cache', 'huggingface'),
        st_home = os.path.join(home_dir, 'st_home') if is_colab() else os.path.join(user_home, '.cache', 'torch', 'sentence_transformers')
    )

    if os.path.exists(config_dir):
        config = read_json(config_dir)
    else:
        config = default_config
        save_json(config, file_dir=config_dir)

    default_keys = list(default_config.keys())
    keys = list(config.keys())

    for k in default_keys:
        if k not in keys:
            config[k] = default_config[k]
            save_json(config, file_dir=config_dir)
    
    for v in config.values():
        if not os.path.exists(v):
            os.makedirs(v)
    if element == 'all':
        return config
    else:
        return config[element]

def set_config(package_home: Optional[str] = None, hf_home: Optional[str] = None, st_home: Optional[str] = None) -> None:
    """Setting paths for the package.

    Args:
        package_home (Optional[str], optional): Home directory for the package if a path is provided. Defaults to None.
        hf_home (Optional[str], optional): Home directory for Huggingface if a path is provided. Defaults to None.
        st_home (Optional[str], optional): Home directory for sentence-transformer if a path is provided. Defaults to None.
    """
    config = get_config()
    if isinstance(package_home, str):
        config['package_home'] = os.path.abspath(package_home)
    if isinstance(hf_home, str):
        config['hf_home'] = os.path.abspath(hf_home)
    if isinstance(st_home, str):
        config['st_home'] = os.path.abspath(st_home)

    save_json(config, get_config_dir())

def validate_type(obj: Any, cls: Any) -> Any:
    """Validate the type of the given object.

    Args:
        obj (Any): Object to validate.
        cls (Any): Class info to validate.

    Returns:
        Any: Th original object if not error is raised.
    """
    if not isinstance(obj, cls):
        raise ValueError(f'{obj} is not a {cls} instance.')
    return obj

