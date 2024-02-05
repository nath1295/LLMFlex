Module llmplus.Tools.web_search_utils
=====================================

Functions
---------

    
`create_content_chunks(contents: List[str], llm: langchain_core.language_models.llms.LLM, chunk_size: int = 400) ‑> List[str]`
:   Create a list of strings of chunks limited by the count of tokens.
    
    Args:
        contents (List[str]): List of contents to aggregate.
        llm (LLM): LLM to count tokens.
        chunk_size (int, optional): Token limit of each chunk. Defaults to 400.
    
    Returns:
        List[str]: List of content chunks.

    
`detect_language(code_snippet: str) ‑> str`
:   Quick guess for the language of the code snippet.
    
    Args:
        code_snippet (str): Code snippet to guess.
    
    Returns:
        str: Programming language.

    
`filtered_child(element: Union[bs4.BeautifulSoup, bs4.element.Tag]) ‑> List[bs4.element.Tag]`
:   Get the filtered list of children of an element.
    
    Args:
        element (Union[BeautifulSoup, Tag]): The element to filter.
    
    Returns:
        List[Tag]: List of children.

    
`format_code(code: bs4.element.Tag, with_wrapper: bool = True) ‑> Optional[str]`
:   

    
`format_header(header: bs4.element.Tag) ‑> str`
:   

    
`format_link(link: bs4.element.Tag) ‑> str`
:   

    
`format_ordered_list(olist: bs4.element.Tag, order: int = 0) ‑> Optional[str]`
:   

    
`format_paragraph(paragraph: bs4.element.Tag) ‑> str`
:   

    
`format_table(table: bs4.element.Tag) ‑> str`
:   

    
`format_unordered_list(ulist: bs4.element.Tag, order: int = 0) ‑> Optional[str]`
:   

    
`get_markdown(url: str, timeout: int = 8, as_list: bool = False) ‑> Union[str, List[str]]`
:   Get the content of a URL as a string or a list of strings.
    
    Args:
        url (str): URL of the website.
        timeout (int, optional): Request timeout as seconds. Defaults to 8.
        as_list (bool, optional): Whether to return the content as a list or as a string. Defaults to False.
    
    Returns:
        Union[str, List[str]]: Content of the URL as a string or a list of string.

    
`get_soup_from_url(url: str, timeout: int = 8) ‑> bs4.BeautifulSoup`
:   Get the soup object from a URL.
    
    Args:
        url (str): URL of the  website.
        timeout (int, optional): Timeout for the request in seconds. Defaults to 8.
    
    Returns:
        BeautifulSoup: Soup object of the website.

    
`process_element(element: Union[bs4.BeautifulSoup, bs4.element.Tag, bs4.element.NavigableString], sep: str = '\n\n', end='  ', as_list: bool = False) ‑> Union[str, List[str], ForwardRef(None)]`
:   Process an element recursively and return the output as text of list of texts by elements.
    
        Args:
            element (Union[BeautifulSoup, Tag, NavigableString]): Element to process.
            sep (str, optional): Seperator of each element. Defaults to '
    
    '.
            end (str, optional): Added string to the end of each element. Defaults to '  '.
            as_list (bool, optional): Whether to return a list of strings of elements or a single string. Defaults to False.
    
        Returns:
            Optional[Union[str, List[str]]]: Content string or list of string of the element.

    
`process_list_children(child: Union[bs4.element.Tag, bs4.element.NavigableString], order: int = 0) ‑> Optional[str]`
:   

    
`process_table_row(row: bs4.element.Tag) ‑> str`
:   

    
`unwanted_contents() ‑> List[str]`
:   Unwanted elements.
    
    Returns:
        List[str]: List of unwanted elements.

    
`wanted_contents() ‑> List[str]`
:   Wanted elements.
    
    Returns:
        List[str]: List of wanted elements.