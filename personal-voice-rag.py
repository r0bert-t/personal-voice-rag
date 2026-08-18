from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import datetime
import gradio as gr


ELEVENLABS_API_KEY = "<KEY>"
voice_response = False

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def toggle_voice_var(value):
    global voice_response
    print(f"[ Updating voice response variable to {value} ]")
    voice_response = value

def run_gradio_ui():
    with gr.Blocks(title="Personal Voice RAG") as app:
        gr.Markdown("""
        # Personal Voice RAG agent
        """)
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record question"
                )
                checkbox = gr.Checkbox(label="Provide voice response", value=voice_response)
                checkbox.change(fn=toggle_voice_var, inputs=checkbox, outputs="")
                submit_btn_audio = gr.Button("Get answer")

            with gr.Column():
                audio_output = gr.Textbox(
                    label="What I heard",
                    interactive=False
                )
                audio_response = gr.Textbox(
                    label="Response",
                    interactive=False
                )

            with gr.Column():
                pdf_file = gr.File(label="Upload a PDF file")
                submit_btn_pdf = gr.Button("Upload file")
                pdf_response = gr.Textbox(
                    label="Response",
                    interactive=False
                )

        with gr.Row():

            with gr.Column():
                chatb = gr.ChatInterface(
                    fn=rag_chain_query,
                    title="Chat",
                    description="Ask me"
                )

        submit_btn_audio.click(fn=transcribe_audio,inputs=[audio_input],outputs=audio_output).then(fn=rag_chain_query,inputs=audio_output,outputs=audio_response)
        submit_btn_pdf.click(fn=pdf_file_ingestion,inputs=[pdf_file],outputs=pdf_response)
                
    app.launch() # share=True

def transcribe_audio(audio_path):
    """ 
    ElevenLabs speech-to-text 
    """

    print("[Transcription] ", end="", flush=True)
    with open(audio_path, "rb") as audio_file:
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

    print("[ Voice response generation ]")

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


def pdf_file_ingestion(file):
    """ 
    Ingest PDF file data into vector database
    """

    # Load the PDF document
    print("[ Loading PDF ]")
    pdf_file = file
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # Split text into manageable chunks
    print('[ Splitting text into chunks ]')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Characters per chunk
        chunk_overlap=200    # Overlap to prevent losing context between chunks
    )
    chunks = text_splitter.split_documents(docs)

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
    resp=is_file_already_indexed(vector_store, pdf_file)
    print('[ File already indexed in vector store? ]', resp)

    if resp:
        return 'File success ingested'
    else:
        return 'Something went wrong'
    

def rag_chain_query(user_query_text, null):
    """ 
    Construct and execute the retrieval-QA chain query
    """

    if not user_query_text.strip():
        print("No voice question")
        return

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

    print("\n[ Processing query ]")
    for chunk in response:
        print(chunk.get("answer",""), end="", flush=True)
        if isinstance(chunk, dict) and "answer" in chunk:
            response_text += chunk["answer"]
            yield response_text
    print("\n")
    
    if voice_response:
        print("[ Converting response to audio ]")
        convert_to_audio(response_text)
    else:
        print("[ No conversion to audio ]")


def main():
    try:
        # Run gradio UI
        run_gradio_ui()
    except:
        print("\nSomething went wrong")

if __name__ == "__main__":
    main()