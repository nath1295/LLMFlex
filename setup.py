from setuptools import setup, find_packages
import os
import subprocess
from typing import Literal, Optional, List

package_name = 'llmplus'

def find_os() -> Literal['Windows', 'Linux', 'MacOS_intel', 'MacOS_apple_silicon', 'MacOS_unknown', 'Unknown']:
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
    try:
        subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        return False

def get_cuda_version() -> Optional[str]:
    import re
    if is_cuda():
        # Run the nvidia-smi command and capture its output
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        
        # Use regular expression to search for the CUDA version in the output
        match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
        if match:
            # Extract and return the CUDA version number
            return match.group(1)
        else:
            return None
    else:
        return None

def get_version() -> str:
    filedir = os.path.join(os.path.dirname(__file__), package_name, '__init__.py')
    with open(filedir, 'r') as f:
        text = f.read()
    import re
    re_version = re.compile(r'__version__[\s]*=[\s]*["\'](\d{1,3}\.\d{1,3}\..+)["\']')
    version = re_version.findall(text)[0]
    return version

def parse_requirements(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        requirements = f.read().splitlines()
    return requirements

# Pre-installation
## pytorch
system = find_os()
cuda = get_cuda_version()
if system == 'Windows':
    if cuda is None:
        os.system('pip3 install torch torchvision torchaudio')
    elif ((cuda >= '11.8') & (cuda < '12.1')):
        os.system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    elif cuda >= '12.1':
        os.system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
    else:
        os.system('pip3 install torch torchvision torchaudio')
elif system == 'Linux':
    if cuda is None:
        os.system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu')
    elif ((cuda >= '11.8') & (cuda < '12.1')):
        os.system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    elif cuda >= '12.1':
        os.system('pip3 install torch torchvision torchaudio')
    else:
        os.system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu')
else:
    os.system('pip3 install torch torchvision torchaudio')

## llama-cpp-python
if cuda is not None:
    os.environ['CMAKE_ARGS'] = "-DLLAMA_CUBLAS=on"
elif system == 'MacOS_apple_silicon':
    os.environ['CMAKE_ARGS'] = '-DLLAMA_METAL=on'
try:
    subprocess.run(["pip", "install", "llama-cpp-python[server]"])
except:
    import warnings
    warnings.warn('Failed to install llama-cpp-python. Please install manually with the guidelines from https://pypi.org/project/llama-cpp-python/.')

# set up package
    
requirements = parse_requirements(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
if is_cuda():
    requirements += parse_requirements(os.path.join(os.path.dirname(__file__), 'requirements_cuda.txt'))

setup(
    name=package_name,
    version=get_version(),
    packages=find_packages(),
    install_requires=requirements,
    entry_points = {
        'console_scripts': [f'{package_name}={package_name}.cli:cli'],
    }
)



    





