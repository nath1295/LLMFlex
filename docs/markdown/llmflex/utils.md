Module llmflex.utils
====================
Utilities for the package.

Functions
---------

`current_time() ‑> float`
:   Getting the current time as a float timestamp, but still human-readable.
    
    Returns:
        float: Timestamp.

`download_file_from_repo(repo_id: str, filename: str, revision: str | None = None, cache_dir: str | None = None, **kwargs) ‑> str`
:   Download a file from a HuggingFace repository.
    
    Args:
        repo_id (str): Repository ID.
        filename (str): Filename.
        revision (Optional[str], optional): Branch of the repository. If None is given, the main branch will be used. Defaults to None.
        cache_dir (Optional[str], optional): Cache directory for saving files from HuggingFace. Defaults to None.
    
    Raises:
        FileExistsError: Raised if the file does not exist in the repository.
    
    Returns:
        str: The local path of the downloaded file.

`env_name() ‑> str`
:   Get the current python environment name.
    
    Returns:
        str: Current python environment name.

`get_config(element: Literal['all', 'package_home', 'hf_home'] = 'all') ‑> Dict[str, str] | str`
:   Get the configuration of the package.
    
    Args: 
        element (Literal[&#39;all&#39;, &#39;package_home&#39;, &#39;hf_home&#39;], optional): The output element of the configuration. Defaults to 'all'.
    
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

`list_repo_files(repo_id: str, revision: str | None = None) ‑> List[str]`
:   List the files in a HuggingFace repository.
    
    Args:
        repo_id (str): Repository ID.
        revision (Optional[str], optional): Branch of the repository. If None is given, the main branch will be used. Defaults to None.
    
    Returns:
        List[str]: List of files in the repository.

`os_name() ‑> Literal['Windows', 'Linux', 'MacOS_intel', 'MacOS_apple_silicon', 'MacOS_unknown', 'Unknown']`
:   Get the current operating system.
    
    Returns:
        Literal['Windows', 'Linux', 'MacOS_intel', 'MacOS_apple_silicon', 'MacOS_unknown', 'Unknown']: The detected OS.

`read_json(file_dir: str) ‑> Dict[str, Any] | List[Dict[str, Any]]`
:   Read the json file provided as a dictionary or a list of dictionaries.
    
    Args:
        file_dir (str): Path of the json file.
    
    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: Content of the json file as a dictionary or a list of dictionaries.

`save_json(content: Dict[str, Any] | List[Dict[str, Any]], file_dir: str) ‑> None`
:   Save the given dictionary or list of dictionaries as a json file.
    
    Args:
        content (Union[Dict[str, Any], List[Dict[str, Any]]]): Dictionary or list of dictionaries to save.
        file_dir (str): Path (with filename) to save the content.

`set_config(package_home: str | None = None, hf_home: str | None = None) ‑> None`
:   Setting paths for the package.
    
    Args:
        package_home (Optional[str], optional): Home directory for the package if a path is provided. Defaults to None.
        hf_home (Optional[str], optional): Home directory for Huggingface if a path is provided. Defaults to None.

`validate_type(obj: Any, cls: Any) ‑> Any`
:   Validate the type of the given object.
    
    Args:
        obj (Any): Object to validate.
        cls (Any): Class info to validate.
    
    Returns:
        Any: Th original object if not error is raised.