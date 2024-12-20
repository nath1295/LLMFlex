from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, List, Optional, Callable

def format_tool_name(s: str) -> str:
    """Format the tool class name into a lower class string.

    Args:
        s (str): The tool class name to be formatted.
    Returns:
        str: The formatted tool class name in lower case.
    """
    import re
    new = re.sub('([A-Z])', '_\\1', s).lower().lstrip('_')
    return re.sub('_+', '_', new).strip('_') 

class ToolOutput(BaseModel):
    content: str
    images: Optional[List[str]] = None
    footnotes: Optional[List[str]] = None

class BaseTool(ABC):
    """Base class for tools.
    """
    @property
    def name(self) -> str:
        """Get the name of the tool.

        This property returns the name of the tool, which is derived from the class name.

        Returns:
            str: The name of the tool.
        """
        if not hasattr(self, '_name'):
            self._name = format_tool_name(self.__class__.__name__)
        return self._name
    
    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        raise NotImplementedError()
    
    def run(self, **kwargs) -> ToolOutput:
        """Run the tool and structure the output into a ToolOutput object. Re-implement it for the subclass if necessary.

        Returns:
            ToolOutput: Structured tool output.
        """
        output = self.__call__(**kwargs)
        if isinstance(output, str):
            pass
        elif isinstance(output, BaseModel):
            output = output.model_dump_json()
        else:
            import json
            try:
                output = json.dumps(output)
            except:
                output = str(output)
        return ToolOutput(content=output)
    
class FunctionTool(BaseTool):
    """Tool wrapper for python functions.
    """
    def __init__(self, fn: Callable) -> None:
        """Initialise the function as a tool.

        Args:
            fn (Callable): The python function.
        """
        from inspect import getdoc
        self._fn = fn
        self._name = fn.__name__
        self.__doc__ = getdoc(fn)

    def __call__(self, **kwargs) -> Any:
        """Call the underneath function.

        Returns:
            Any: Output of the function.
        """
        return self._fn(**kwargs)
