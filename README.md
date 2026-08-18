# Personal Voice RAG agent

- Supports spoken audio to interact with AI models with voice high accuracy, ultra-low latency and deep multilingual support 
- Allows to perform fast semantic search over large sets of private documents in a local vector database
- Eliminate token costs and allows to run large language models (LLMs) directly on user own hardware without relying on external servers

___
> **Use-cases:**
> Personal voice agent is able to provide more accurate, context-aware answers by matching a user's query to information stored in a local vector database ([semantic search](https://en.wikipedia.org/wiki/Semantic_search)).
> It can be used as a powerful AI assistant by connecting a LLMs to private documentation without a need to retrain the model.
___

![Python](https://img.shields.io/badge/Python-grey)
![Ollama](https://img.shields.io/badge/Ollama-orange)
![Chroma](https://img.shields.io/badge/Chroma-yellow)
![ElevenLabs](https://img.shields.io/badge/ElevenLabsAPI-black)


![System architecture](https://github.com/r0bert-t/personal-voice-rag/blob/main/docs/personal-voice-rag.png)


## System Architecture

Personal voice agent is using [Ollama](https://ollama.com/) as the core LLM engine, open-source [LangChain](https://www.langchain.com/) development framework that acts as a bridge between the AI model and data sources and [Gradio](https://gradio.app/) to provide a Web based UI to the user. To store user data (e.g. PDF files) it was used [Chroma](https://www.trychroma.com/) vector database . 

### Gradio
Gradio is an open-source Python package that allows to quickly access the personal voice agent using a web interface.

### Ollama
Ollama is a free, open-source software platform that allows you to run, manage, and deploy large language models directly on local computer.

### LLM model
Meta's [Llama 3.2 (3B)](https://ollama.com/library/llama3.2:3b) is a lightweight text model that runs smoothly on almost any modern personal computer and achieves fast response speeds on standard CPUs, Apple Silicon or dedicated graphics cards. It offers maximum context window of 128K tokens.
Of course it is possible to use other models available in [Ollama library](https://ollama.com/library) depending on the available hardware performance.

### Embedding model 
For embedding it was used a [nomic-embed-text:v1.5](https://ollama.com/library/nomic-embed-text) which is a high-performing open embedding model with a large token context window.
It converts text into dense numerical vectors that capture semantic meaning and also allows to search local vector store for relevant context.

### Chroma database
Chroma database (ChromaDB) is an open-source vector store used for storing and retrieving vector embeddings. It supports seamless connectivity with LangChain framework for RAG pipelines.

### Spoken audio support

Personal voice agent is supporting following ElevenLabs APIs:
- **ElevenLabs Speech to Text (STT)** powered by the [Scribe v2](https://elevenlabs.io/speech-to-text) model, that converts spoken audio into highly accurate written text. It can handle complex audio environments (e.g. with background noise, overlapping speech).
- **ElevenLabs Text-to-Speech (TTS)** that enables to generate lifelike, emotionally rich, and human-like speech from text in over 70 language. It features ultra-low latency and supports real-time audio streaming. Personal voice agent was configured to use for text to audio conversion [Eleven v3](https://elevenlabs.io/v3) model which is most expressive ElevenLabs AI voice model

> Please note that in order to use ElevenLabs APIs you have to generate a valid API key on [ElevenLabs Platform](https://elevenlabs.io/api) and set the proper endpoints access

### RAG chain pipeline
It is worth to outline a few important steps in processing RAG chain and user queries
1. Convert a vector store into a retriever object (LangChain retriever interface) using **as_retriever** function. In the code we limit the search to return only the top 3 (default is 4) most relevant document chunks for any given query.
2. Build a structured multi-role chat prompt using a LangChain core method **ChatPromptTemplate.from_messages**.
In code we are using following system prompt that instructs the model what to do:
```
  system_prompt = (
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, check the LLM model.
   )
```
3. Build a runnable sequence using a **create_stuff_documents_chain** function. This fucntion "stuffs" a list of retrieved documents into a single prompt context window, formats them, and sends them to a large language model.
4. Combine a retriever and a document combination chain using a **create_retrieval_chain** function
5. Execute a RAG pipeline to the Large Language Model (LLM) and stream response in real-time to user

### Sample data flow (voice user query)
1.	Agent records user voice query and converts it to text using ElevenLabs SST API
2.  The agent uses an embedding model to search local vector store for relevant context (e.g. indexed PDF files)
2.  The retrieved query (in text format) and context is injected into the LLM prompt
3.	Llama 3.2 model synthesizes an accurate answer based on provided data
4.	The retrieved response is converted to audio using ElevenLabs TTS API
5.	ElevenLabs reads it to the user

### Voice RAG setup
```bash
# Install Ollama (Mac/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Install Llama3.2 LLM model
ollama pull llama3.2:3b

# Install nomic-embed-text model
ollama pull nomic-embed-text

# Clone repository
git clone https://github.com/r0bert-t/personal-voice-rag.git
cd personal-voice-rag

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Update ElevenLabs API key
export ELEVENLABS_API_KEY="<KEY>”

# Run code
python personal-voice-rag.py

# Open in web browser following link to access UI
http://localhost:7860
```
## Privacy considerations 

When using a speech to interact with AI agent, some requests are sent to ElevenLabs platform. ElevenLabs offers [Zero Retention Mode](https://elevenlabs.io/docs/eleven-api/resources/zero-retention-mode) that can be enabled for STT and TTS APIs, when most data in requests and responses are immediately deleted once the request is completed, however it is limited only to enterprise customers.
When we want to ensure a full communication privacy it is recommended to use a text interface to interact with AI and keep all queries processing in the local-hosted RAG and private LLM endpoints.

## Cost
| Service       | Cost |
|---------------|------|
| Ollama        | Free (local) |
| ChromaDB      | Free (local) |
| Llama 3.2(3b) | Free (local) |
| nomic-embed-text | Free (local) |
| ElevenLabsAPI | Free tier 10000 credits/month |

## License

MIT

---

Created by [Robert Tracz](https://www.linkedin.com/in/robert-tracz/) 