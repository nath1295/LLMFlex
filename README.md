# LLMPlus

LLMPlus is a python package that allows python developers to work with open-source large language models (LLM) and do prompt engineering with a simple interface.

It provides classes to load LLM models, embedding models, and vector databases to create LLM powered applications with your own prompt engineering and RAG techniques. With a one-liner command, you can load to chatbot interface to chat with the LLM as well.  

## Prerequisites

Before you begin, your python version is >= 3.9. Creating a virtual python environment with conda is before installing the package.

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
