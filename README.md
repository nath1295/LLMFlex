# LLMPlus

LLMPlus is a python package that allows python developers to work with different large language models (LLM) and do prompt engineering with a simple interface. It favours free and local resources instead of using paid APIs to develop truely local and private AI-powered solutions.

It provides classes to load LLM models, embedding models, and vector databases to create LLM powered applications with your own prompt engineering and RAG techniques. With a one-liner command, you can load to chatbot interface to chat with the LLM or serve a model as OpenAI API as well.  

## Features
### 1. Multilple LLMs with different genration configurations from one model 
Unlike Langchain, you can create multiple llms with different temperature, max new tokens, stop words etc. with the same underlying model without loading the model several times using the `LlmFactory` class. This can be useful when you create your own agent with different llm tasks which requires different configurations.

### 2. Langchain compatibility with enhanced performances
All the llms created with `LlmFactory` are langchain compatible, and can be seamlessly integrated in your existing Langchain code. All the llm classes are re-implementations of some langchain llm classes which support more efficient streaming and stop words management, all with a unified interface.

### 3. Multiple model formats support
Torch, AWQ, GPTQ, and GGUF models, and OpenAI API are all supported, and the loading process are all handled in the `LlmFactory` class, so it is just plug and play. 

### 4. Embedding Toolkits
Bundled classes for using embedding models which contains the embedding model and a tokens-count-based text splitter using the embedding model.

### 5. Vector database
Utilising Embedding toolkits and FAISS, a `VectorDatabase` class can allow you to store and search texts for your RAG tasks.

### 6. Chat memories
Chat memory classes for storing chat memory on disk.  
1. `BaseConversationMemory`  
Memory class without using any embedding models or vector databases.  

2. `LongShortConversationMemory`  
Memory class using an underlying `VectorDatabase` to maintain long term memory along with the most recent memory.

### 7. Prompt template
A `PromptTemplate` class is implemented to format your prompt with different prompt formats for models from different sources. Some presets like `Llama2`, `ChatML`, `Vicuna`, and more are already implemented, but you can alway add your own prompt format template.

### 8. Custom tools
A base class `BaseTool` for creating llm powered tools. A `WebSearchTool` powered by __DuckDuckGo__ is implemented as an example.

### 9. Chatbot frontend interface
If you simply want to play with a model, there's a gradio frontend chatbot that allows you to chat with a model with different generation configurations. You can switch between chat histories and prompt format, and you can set your system prompt and other model text generation sampling configurations in the gradio webapp.

## Prerequisites

Before you begin, make sure your python version is >= 3.9. Creating a virtual python environment with conda is highly recommended before installing the package.

## Installing LLMPlus

You can install LLMPlus with pip easily.

```
pip install git+https://github.com/nath1295/LLMPlus.git
```

## Using LLMPlus

This is how you can start with any text generation model on HuggingFace with your computer.

```python
from llmplus import LlmFactory

# Load the model from Huggingface
model = LlmFactory('TheBloke/OpenHermes-2.5-Mistral-7B-GGUF')

# Create a llm
llm = model(temperature=0.7, max_new_tokens=512, stop=['```'])

# Use the LLM for your task
prompt = """Instruction:
Write a python script to load a Pandas Dataframe from the csv file 'test.csv'

Solution:
```python
"""
script = llm(prompt)
print(script)

# Or if you prefer to generate the output with token streamming.
for token in llm.stream(prompt):
    print(token, end='')
```

To load an embedding model and use a vector database:

```python
from llmplus import HuggingfaceEmbeddingsToolkit, VectorDatabase

# Loading the embedding model toolkit
embeddings = HuggingfaceEmbeddingsToolkit(model_id='thenlper/gte-large')

# Create a vector database
food = ['Apple', 'Banana', 'Pork']
vectordb = VectorDatabase.from_data(index=food, embeddings=embeddings)

# Do semantic search on the vector database
print(vectordb.search('Beef'))
```

Or if you just want a GUI to start chatting with your LLM model with both long term and short term memory, type this command in the terminal:
```bash
llmplus interface --model_id TheBloke/OpenHermes-2.5-Mistral-7B-GGUF --embeddings thenlper/gte-large
```
You will see a gradio frontend, use it to chat with the LLM model.  
![Gradio GUI](imgs/chat_gui.png)

## Documentations
Python documentation for all the classes, methods, and functions is provided in the `./docs` directory in this repository.

## License

This project is licensed under the terms of the MIT license.
