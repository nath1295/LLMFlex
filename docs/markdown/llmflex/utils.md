Module llmflex.utils
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

    
`get_config(element: Literal['all', 'package_home', 'hf_home', 'st_home'] = 'all') ‑> Union[Dict[str, str], str]`
:   Get the configuration of the package.
    
    Args: 
        element (Literal[&#39;all&#39;, &#39;package_home&#39;, &#39;hf_home&#39;, &#39;st_home&#39;], optional): The output element of the configuration. Defaults to 'all'.
    
    Returns:
        Union[Dict[str, str], str]: Configuration of the package or one of hte configured directories.

    
`get_config_dir() ‑> str`
:   Get the configuration file path of the package.
    
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

    
`set_config(package_home: Optional[str] = None, hf_home: Optional[str] = None, st_home: Optional[str] = None) ‑> None`
:   Setting paths for the package.
    
    Args:
        package_home (Optional[str], optional): Home directory for the package if a path is provided. Defaults to None.
        hf_home (Optional[str], optional): Home directory for Huggingface if a path is provided. Defaults to None.
        st_home (Optional[str], optional): Home directory for sentence-transformer if a path is provided. Defaults to None.

    
`validate_type(obj: Any, cls: Any) ‑> Any`
:   Validate the type of the given object.
    
    Args:
        obj (Any): Object to validate.
        cls (Any): Class info to validate.
    
    Returns:
        Any: Th original object if not error is raised.