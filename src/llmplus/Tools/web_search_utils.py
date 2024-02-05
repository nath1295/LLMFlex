from bs4 import BeautifulSoup, NavigableString, Tag
from langchain.llms.base import LLM
from typing import Optional, List, Union

def get_soup_from_url(url: str, timeout: int = 8) -> BeautifulSoup:
    """Get the soup object from a URL.

    Args:
        url (str): URL of the  website.
        timeout (int, optional): Timeout for the request in seconds. Defaults to 8.

    Returns:
        BeautifulSoup: Soup object of the website.
    """
    import requests
    from fake_useragent import UserAgent
    agent  = UserAgent(os = ['windows', 'macos'])
    response = requests.get(url, headers={'User-agent': agent.random}, timeout=timeout)
    if response.status_code != 200:
        return BeautifulSoup('', 'html.parser')
    return BeautifulSoup(response.content, 'html.parser')

def unwanted_contents() -> List[str]:
    """Unwanted elements.

    Returns:
        List[str]: List of unwanted elements.
    """
    unwanted = ['notification-bar', 'banner', 'nav', 'footer', 'sidebar', '.nav', '.footer', '.sidebar', '#nav', '#footer', '#sidebar']
    return unwanted

def wanted_contents() -> List[str]:
    """Wanted elements.

    Returns:
        List[str]: List of wanted elements.
    """
    wanted = ['a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table', 'article', 'section', 'blockquote', 'code', 'pre', 'samp']
    return wanted

def filtered_child(element: Union[BeautifulSoup, Tag]) -> List[Tag]:
    """Get the filtered list of children of an element.

    Args:
        element (Union[BeautifulSoup, Tag]): The element to filter.

    Returns:
        List[Tag]: List of children.
    """
    children = element.children
    output = []
    for child in children:
        if isinstance(child, NavigableString):
            output.append(child)
        elif child.name in unwanted_contents():
            pass
        elif any(c in unwanted_contents() for c in child.get('class', [])):
            pass
        elif child.get('id', '') in unwanted_contents():
            pass
        else:
            output.append(child)
    return output
    
def process_table_row(row: Tag) -> str:
    children = list(row.children)
    children = list(filter(lambda x: x.name in ['th', 'td'], children))
    if len(children) == 0:
        return None
    is_header = children[0].name == 'th'
    data = []
    for child in children:
        out = list(map(process_list_children, child.children))
        out = list(filter(lambda x: x is not None, out))
        out = ' '.join(out)
        out = 'EMPTY CELL' if out.strip(' \n\r\t')=='' else out
        data.append(out)
    num = len(data)
    data = '| ' + ' | '.join(data) + ' |'
    if is_header:
        seps = [':---'] * num
        seps = '| ' + ' | '.join(seps) + ' |'
        data += '\n' + seps
    return data

def format_table(table: Tag) -> str:
    children = list(filter(lambda x: x.name == 'tr', list(table.children)))
    children = list(map(process_table_row, children))
    children = list(filter(lambda x: x is not None, children))
    if len(children) == 0:
        return None
    return '\n'.join(children)

def process_list_children(child: Union[Tag, NavigableString], order: int = 0) -> Optional[str]:
    if isinstance(child, NavigableString):
        out = child.get_text(strip=True)
        out = None if out.strip(' \n\r\t') == '' else out
    elif child.name == 'a':
        out = format_link(child)
    elif child.name =='ol':
        out = format_ordered_list(child, order=order + 1)
    elif child.name =='ul':
        out = format_unordered_list(child, order=order + 1)
    else:
        out = child.get_text(strip=True)
        out = None if out.strip(' \n\r\t') == '' else out
    return out

def format_ordered_list(olist: Tag, order: int = 0) -> Optional[str]:
    count = 0
    outputs = []
    for l in olist.children:
        if not isinstance(l, Tag):
            continue
        elif l.name != 'li':
            continue
        else:
            child = list(map(lambda x: process_list_children(x, order), list(l.children)))
            child = list(filter(lambda x: x is not None, child))
            if len(child) == 0: 
                out = None
            else:
                out = ' '.join(child)
            if out is None:
                continue
            if out.strip(' \n\r\t') == '':
                continue
            else:
                count += 1
                outputs.append('\t' * order + f'{count}. {out}')

    if len(outputs) == 0:
        return None
    else:
        return '\n'.join(outputs)

def format_unordered_list(ulist: Tag, order: int = 0) -> Optional[str]:
    outputs = []
    for l in ulist.children:
        if not isinstance(l, Tag):
            continue
        elif l.name != 'li':
            continue
        else:
            child = list(map(lambda x: process_list_children(x, order), list(l.children)))
            child = list(filter(lambda x: x is not None, child))
            if len(child) == 0: 
                out = None
            else:
                out = ' '.join(child)
            if out is None:
                continue
            if out.strip(' \n\r\t') == '':
                continue
            else:
                outputs.append('\t' * order + f'* {out}')

    if len(outputs) == 0:
        return None
    else:
        return '\n'.join(outputs)

def detect_language(code_snippet: str) -> str:
    """Quick guess for the language of the code snippet.

    Args:
        code_snippet (str): Code snippet to guess.

    Returns:
        str: Programming language.
    """
    # Normalize the code snippet to help with detection
    code_snippet_lower = code_snippet.lower()

    if 'class' in code_snippet_lower and 'public static void main' in code_snippet:
        return 'java'
    elif ('def ' in code_snippet or 'import ' in code_snippet) and ':' in code_snippet:
        return 'python'
    elif ('function ' in code_snippet or '=>' in code_snippet) and ('var ' in code_snippet or 'let ' in code_snippet or 'const ' in code_snippet):
        return 'javascript'
    elif '#include' in code_snippet:
        return 'cpp'
    elif code_snippet.startswith('#!/bin/bash') or 'echo ' in code_snippet or 'grep ' in code_snippet:
        return 'bash'
    elif 'def ' in code_snippet and 'end' in code_snippet:
        return 'ruby'
    elif '<?php' in code_snippet_lower or 'echo ' in code_snippet or '->' in code_snippet:
        return 'php'
    elif 'using ' in code_snippet and 'namespace ' in code_snippet:
        return 'csharp'  # Note: Markdown typically uses 'cs' or 'csharp' for C#
    elif '<html>' in code_snippet_lower or '<div>' in code_snippet_lower or 'doctype html' in code_snippet_lower:
        return 'html'
    elif '{' in code_snippet and '}' in code_snippet and (':' in code_snippet or ';' in code_snippet) and ('color:' in code_snippet_lower or 'background:' in code_snippet_lower or 'font-size:' in code_snippet_lower):
        return 'css'
    else:
        return 'plaintext'  # Using 'plaintext' for unknown or plain text code blocks

def format_code(code: Tag, with_wrapper: bool = True) -> Optional[str]:
    text = code.get_text(strip=True)
    if text.strip(' \n\r\t') =='':
        return None
    else:
        output = text.strip(' \n\r\t')
    if with_wrapper:
        return f'```{detect_language(output)}\n' + output + '\n```'
    
def format_paragraph(paragraph: Tag) -> str:
    outputs = []
    for child in filtered_child(paragraph):
        if isinstance(child, NavigableString):
            outputs.append(child.get_text(strip=True))
        elif child.name in ['pre', 'code', 'samp']:
            code = format_code(child, with_wrapper=False)
            if code is not None:
                outputs.append(f'`{code}`')
            else:
                code = child.get_text(strip=True)
                outputs.append(f'`{code}`')

        elif child.name == 'a':
            outputs.append(format_link(child))
        else:
            outputs.append(child.get_text(strip=True))
    if len(outputs) == 0:
        return None
    else:
        return ' '.join(outputs)

def format_header(header: Tag) -> str:
    size = int(header.name[1])
    return '#' * size + ' ' + header.get_text(strip=True)

def format_link(link: Tag) -> str:
    text = link.get_text(strip=True)
    href = link.get('href', '')
    if ((text.strip(' \n\r\t#') == '') & (href.strip(' \n\r\t#') == '')):
        return None
    elif text.strip(' \n\r\t') == '':
        output = f'[{href}]'
    elif href.strip(' \n\r\t') == '':
        output = text
    else:
        output = f'[{text}]({href})'
    return output

def process_element(element: Union[BeautifulSoup, Tag, NavigableString], sep: str = '\n\n', end='  ', as_list: bool = False) -> Optional[Union[str, List[str]]]:
    """Process an element recursively and return the output as text of list of texts by elements.

    Args:
        element (Union[BeautifulSoup, Tag, NavigableString]): Element to process.
        sep (str, optional): Seperator of each element. Defaults to '\n\n'.
        end (str, optional): Added string to the end of each element. Defaults to '  '.
        as_list (bool, optional): Whether to return a list of strings of elements or a single string. Defaults to False.

    Returns:
        Optional[Union[str, List[str]]]: Content string or list of string of the element.
    """
    outputs = []
    for e in filtered_child(element):
        if isinstance(e, NavigableString):
            text = e.get_text(strip=True)
            text = None if text.strip(' \n\r\t') == '' else text
            outputs.append(text)
        elif e.name in ['pre', 'code', 'samp']:
            outputs.append(format_code(e))
        elif e.name == 'ul':
            outputs.append(format_unordered_list(e))
        elif e.name == 'ol':
            outputs.append(format_ordered_list(e))
        elif e.name == 'table':
            outputs.append(format_table(e))
        elif e.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            outputs.append(format_header(e))
        elif e.name == 'a':
            outputs.append(format_link(e))
        elif e.name in ['p']:
            outputs.append(format_paragraph(e))
        else:
            outputs.append(process_element(e, as_list=as_list))
    final = []
    for o in outputs:
        if o is None:
            continue
        elif isinstance(o, list):
            final.extend(o)
        elif o.strip(' \n\r\t') == '':
            continue
        elif len(o) < 3: # Remove random elements with less than 3 characters.
            continue
        else:
            final.append(o)
    if len(final) == 0:
        return None
    if as_list:
        return final
    else:
        final = list(map(lambda x: x + end, final))
        return sep.join(final)
    
def create_content_chunks(contents: List[str], llm: LLM, chunk_size: int = 400) -> List[str]:
    """Create a list of strings of chunks limited by the count of tokens.

    Args:
        contents (List[str]): List of contents to aggregate.
        llm (LLM): LLM to count tokens.
        chunk_size (int, optional): Token limit of each chunk. Defaults to 400.

    Returns:
        List[str]: List of content chunks.
    """
    chunks = []
    current = []
    current_count = 0
    for c in contents:
        count = llm.get_num_tokens(c)
        if current_count + count <= chunk_size:
            current.append(c)
            current_count += count
        elif count > chunk_size:
            chunks.append('\n\n'.join(current))
            chunks.append(c)
            current = []
            current_count = 0
        else:
            chunks.append('\n\n'.join(current))
            current = [c]
            current_count = count
    if len(current) != 0:
        chunks.append('\n\n'.join(current))
    return chunks

def get_markdown(url: str, timeout: int = 8, as_list: bool = False) -> Union[str, List[str]]:
    """Get the content of a URL as a string or a list of strings.

    Args:
        url (str): URL of the website.
        timeout (int, optional): Request timeout as seconds. Defaults to 8.
        as_list (bool, optional): Whether to return the content as a list or as a string. Defaults to False.

    Returns:
        Union[str, List[str]]: Content of the URL as a string or a list of string.
    """
    soup = get_soup_from_url(url, timeout=timeout)
    return process_element(soup, as_list=as_list)
