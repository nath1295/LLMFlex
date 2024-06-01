from .tool_utils import direct_response
from .browser_tool import BrowserTool
from .sd_tool import StableDiffusionTool

def math_tool(equation: str) -> float:
    """Used for doing maths task, return the answer of the given maths equation.

    Args:
        equation (str): Maths equation that can be interpret by python using `eval()`.

    Returns:
        float: The answer of the maths equation.
    """
    return eval(equation)