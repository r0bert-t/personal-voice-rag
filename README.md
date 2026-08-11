# Personal Voice RAG agent

- Supports spoken audio to interact with AI models with voice high accuracy, ultra-low latency and deep multilingual support 
- Allows to perform fast semantic search over large sets of private documents in a local vector database
- Eliminate token costs and allows to run large language models (LLMs) directly on user own hardware without relying on external servers

___
> **Use-cases:**
> Personal voice agent is able to provide more accurate, context-aware answers by matching a user's query to information stored in a local vector database (semantic search).
> It can be used as a powerful AI assistant by connecting a LLMs to private documentation without needing to retrain the model.
___

![Python](https://img.shields.io/badge/Python-grey)
![Ollama](https://img.shields.io/badge/Ollama-orange)
![Chroma](https://img.shields.io/badge/Chroma-yellow)
![ElevenLabs](https://img.shields.io/badge/ElevenLabsAPI-black)


![System architecture](https://github.com/r0bert-t/personal-voice-rag/blob/main/docs/personal-voice-rag.png)


## System Architecture

Personal voice agent is using [Ollama](https://ollama.com/) as the core engine and [Chroma](https://www.trychroma.com/) vector database for storing user data. Ollama is a free, open-source software platform that allows you to run, manage, and deploy large language models directly on local computer.
Chroma database (ChromaDB) is an open-source vector store used for storing and retrieving vector embeddings.

### LLM model
For LLM is was used [Llama 3.2 (3B)](https://ollama.com/library/llama3.2:3b) model that balances high-performance text generation with ultra-low hardware requirements and it is perfectly suited for hardware like personal computers. It offers maximum context window of 128K tokens

### Embedding model 
For embedding it was used a [nomic-embed-text](https://ollama.com/library/nomic-embed-text) which is a high-performing open embedding model with a large token context window
It converts text into dense numerical vectors that capture semantic meaning, making it useful for retrieval-augmented generation (RAG) and semantic search.

### Spoken audio support

Personal voice agent is supporting following ElevenLabs APIs:
- **ElevenLabs Speech to Text (STT)** powered by the [Scribe v2](https://elevenlabs.io/speech-to-text) model, that converts spoken audio into highly accurate written text. It can handle complex audio environments (e.g. with background noise, overlapping speech).
- **ElevenLabs Text-to-Speech (TTS)** that enables developers to generate lifelike, emotionally rich, and human-like speech from text in over 70 language. It features ultra-low latency and supports real-time audio streaming. In our text conversion to audio we will be using [Eleven v3](https://elevenlabs.io/v3) model which is most expressive AI voice model

### Sample data flow
1.	Agent records user voice query and converts it to text using ElevenLabs SST API
2.  The agent uses an embedding model to search local vector store for relevant context (e.g. PDF files)
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

# Create environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install libportaudio
sudo apt install libportaudio

# Update ElevenLabs API key
export ELEVENLABS_API_KEY="<KEY>”

# Run code
python personal-voice-rag.py
```

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