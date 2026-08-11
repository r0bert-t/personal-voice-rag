"""
Personal Voice RAG agent

Description: Allows to interact with AI models hosted on local Ollama using a speech and to perform fast semantic search
over large sets of private documents in a local vector database.
Version: 0.0.1
"""

import os
import queue
import sys
import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from scipy.io.wavfile import write
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import datetime


ELEVENLABS_API_KEY = "<KEY>"
SAMPLE_RATE = 16000
FILENAME = "input.wav"


client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def record_audio():
    """ 
    Audio recording 
    """

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("\n[Speak] Press ENTER to end recording", end="", flush=True)
    
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
    with stream:
        input()

    # Audio data collection
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())
    
    audio_np = np.concatenate(audio_data, axis=0)
    write(FILENAME, SAMPLE_RATE, audio_np)
    print("[Voice recorded]")

def transcribe_audio():
    """ 
    ElevenLabs speech-to-text 
    """

    print("[Transcription] ", end="", flush=True)
    with open(FILENAME, "rb") as audio_file:
        transcription = client.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v2",
        )
    print(f"Audio transcription: {transcription.text}")
    return transcription.text

def convert_to_audio(text):
    """ 
    ElevenLabs text-to-speech 
    """

    print("[Voice response generation]")

    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_v3",
        output_format="mp3_44100_128",
    )

    play(audio_stream)

def is_file_already_indexed(vector_store, file_name):
    """
    Checks if the specified file path already exists in the vector store's metadata.
    """

    # Check Chroma database to get records matching a specific metadata filter
    existing_doc = vector_store.get(where={"source": file_name}, limit=1)
    
    # If the 'ids' list contains elements, the file has already been indexed
    return len(existing_doc['ids']) > 0


def pdf_file_ingestion():
    """ 
    Ingest PDF file data into vector database
    """

    # Load the PDF document
    print("[ Loading PDF ]")
    pdf_file = input('Provide file name: ')
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # Split text into manageable chunks
    print('[ Splitting text into chunks ]')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Characters per chunk
        chunk_overlap=200    # Overlap to prevent losing context between chunks
    )
    chunks = text_splitter.split_documents(docs)
    #print(chunks)

    # Initialize local Ollama Embeddings
    print('[ Initializing local embeddings ]')
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"  # High-performance local embedding model
    )

    # Create local Vector store (Chroma) and index the chunks
    print('[ Indexing chunks into Chroma vector store ]')

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Check if file is already indexed in Chroma database
    print('[ File already indexed in vector store? ]', is_file_already_indexed(vector_store, pdf_file))


def rag_chain_query(query_format):
    """ 
    Construct and execute the retrieval-QA chain query
    """

    if query_format == 'voice':
        flag = input('\nProceed with voice question? [y/n] ').strip().lower()
        if flag == 'y':
            record_audio()
            user_query_text = transcribe_audio()

        if not user_query_text.strip():
            print("No voice question")
            return
    elif query_format == 'text':
        user_query_text = input('Provide question: ').strip()


    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    
    # Create the RAG prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, check the LLM model.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Build and execute the retrieval-QA chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.stream({"input": user_query_text})

    response_text = ''

    print("\n[Processing query]")
    for chunk in response:
        print(chunk.get("answer",""), end="", flush=True)
        response_text += chunk
    print("\n")
    
    # Convert response to audio
    convert_to_audio(response_text)
    
    # Remove temporary file
    if os.path.exists(FILENAME):
        os.remove(FILENAME)

def main():
    try:
        flag_pdf_ingestion = input('\nDo you want to ingest PDF file? [y/n] ').strip().lower()
        if flag_pdf_ingestion == 'y':
            pdf_file_ingestion()
        while True:
            flag_query = input('\nProvide type of the question? [text/voice] ').strip().lower()
            if flag_query == 'text' or flag_query == 'voice':
                rag_chain_query(flag_query)
            else:
                break      
    except KeyboardInterrupt:
        print("\nSession has been ended")

if __name__ == "__main__":
    main()