Module llmplus.utils
====================

Functions
---------

    
`current_time() ‑> float`
:   Getting the current time as a float timestamp, but still human-readable.
    
    Returns:
        float: Timestamp.

    
`env_name() ‑> str`
:   Get the current python environment name.
    
    Returns:
        str: Current python environment name.

    
`get_config() ‑> Dict[str, Any]`
:   Get the configuration of llmplus.
    
    Returns:
        Dict[str, Any]: Configuration of llmplus.

    
`get_config_dir() ‑> str`
:   Get the configuration file path of llmplus.
    
    Returns:
        str: configuration file path.

    
`is_colab() ‑> bool`
:   Check if it's on Google Colab.
    Returns:
        bool: True if on Google Colab.

    
`is_conda() ‑> bool`
:   Check if it's in a conda environment
    
    Returns:
        bool: True if in a conda environment.

    
`is_cuda() ‑> bool`
:   Whether CUDA is available.
    
    Returns:
        bool: True if CUDA is available.

    
`os_name() ‑> Literal['Windows', 'Linux', 'MacOS_intel', 'MacOS_apple_silicon', 'MacOS_unknown', 'Unknown']`
:   Get the current operating system.
    
    Returns:
        Literal['Windows', 'Linux', 'MacOS_intel', 'MacOS_apple_silicon', 'MacOS_unknown', 'Unknown']: The detected OS.

    
`read_json(file_dir: str) ‑> Union[Dict[str, Any], List[Dict[str, Any]]]`
:   Read the json file provided as a dictionary or a list of dictionaries.
    
    Args:
        file_dir (str): Path of the json file.
    
    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: Content of the json file as a dictionary or a list of dictionaries.

    
`save_json(content: Union[Dict[str, Any], List[Dict[str, Any]]], file_dir: str) ‑> None`
:   Save the given dictionary or list of dictionaries as a json file.
    
    Args:
        content (Union[Dict[str, Any], List[Dict[str, Any]]]): Dictionary or list of dictionaries to save.
        file_dir (str): Path (with filename) to save the content.

    
`set_config(llmplus_home: Optional[str] = None, hf_home: Optional[str] = None, st_home: Optional[str] = None) ‑> None`
:   Setting paths for llmplus.
    
    Args:
        llmplus_home (Optional[str], optional): Home directory for llmplus if a path is provided. Defaults to None.
        hf_home (Optional[str], optional): Home directory for Huggingface if a path is provided. Defaults to None.
        st_home (Optional[str], optional): Home directory for sentence-transformer if a path is provided. Defaults to None.