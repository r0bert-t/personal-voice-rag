"""
Personal Voice RAG agent

Description: Allows to interact with AI models hosted on local Ollama using a text/speech and to perform fast semantic search
over large sets of private documents in a local vector database.

Version: 0.0.7
"""

import sys
import warnings


def custom_show_warning(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = custom_show_warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import gradio as gr
from fastmcp import FastMCP
import chromadb
import asyncio
import os

ELEVENLABS_API_KEY = "<KEY>"

# Expose Gradio UI using a public endpoint (Internet)
gradio_public_endpoint = False

# Run MCP server
mcp_server = True

# Location of the vector database
current_dir = os.path.dirname(os.path.abspath(__file__))
vectordb_path = os.path.join(current_dir, "chroma_db")

os.makedirs(vectordb_path, exist_ok=True)

# Initialize the persistent ChromaDB client
persistent_client = chromadb.PersistentClient(
    path=vectordb_path,
    settings=chromadb.Settings(anonymized_telemetry=False)
)

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
mcp = FastMCP("MCP server")

voice_response = False
mq_retriever = False


def toggle_global_vars(var_name, value):
    """
    Update global variables
    """
    global voice_response
    global mq_retriever
    if var_name == "voice_response":
        print(f"[ Updating voice response variable to {value} ]")
        voice_response = value
    elif var_name == "mq_retriever":
        print(f"[ Updating multi query retriever variable to {value} ]")
        mq_retriever = value


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
        model_id="eleven_v4_turbo",
        output_format="mp3_44100_128",
    )

    play(audio_stream)


def get_existing_collections():
    """
    Fetches collection names from ChromaDB
    """

    try:
        # Return a list of all collection and bypass the default pagination limit
        collections = persistent_client.list_collections(limit=10000)

        # Extract names and exclude any blank spaces and transient errors
        names = sorted([col.name for col in collections if col.name.strip()])

        return names
    except Exception as e:
        print(f"Vector database error during collections fetch: {str(e)}")
        return []


def refresh_collections_dropdown():
    """
    Refresh collection list
    """

    updated_list = get_existing_collections()

    if not updated_list:
        return gr.Dropdown(choices=["No collections found"], value="No collections found")

    return gr.Dropdown(choices=updated_list, value=updated_list[0])


def check_indexed_files(collection):
    """
    Provide a list of indexed files retrieved from vector store's metadata
    """

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma(
        embedding_function=embeddings,
        client=persistent_client,
        collection_name=collection
    )

    # Fetch data and request only metadatas
    data = vector_store.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])

    indexed_files = {}

    # Use a set to keep track of uniquely seen files
    seen_files = set()

    # Iterate directly over the retrieved metadata list
    if metadatas:
        for doc in metadatas:
            source = doc.get("source", "Unknown Source")
            creation_date = doc.get("creationdate", "Unknown Date")

            # Extract filename from the path
            filename = os.path.basename(source)

            # Create a unique identifier for the file
            file_fingerprint = (source, creation_date)

            if file_fingerprint not in seen_files:
                # print(f"Source: {filename} | Created: {creation_date}")
                seen_files.add(file_fingerprint)
                indexed_files[filename] = creation_date

    else:
        return "No indexed documents or metadata found."

    files_list = [item for item in indexed_files.keys()]
    return "\n".join(files_list)


def is_file_already_indexed(vector_store, file_name):
    """
    Checks if the specified file path already exists in the vector store's metadata.
    """

    # Check Chroma database to get records matching a specific metadata filter
    existing_doc = vector_store.get(where={"source": file_name}, limit=1)

    # If the 'ids' list contains elements, the file has already been indexed
    return len(existing_doc['ids']) > 0


def pdf_file_ingestion(file, collection_input):
    """
    Ingest PDF file data into vector database
    """

    if not file:
        return 'Provide a valid file'

    if not collection_input:
        return 'Provide a collection name'

    # Load the PDF document
    print("[ Loading PDF ]")
    pdf_file = file
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # Split text into manageable chunks
    print('[ Splitting text into chunks ]')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Characters per chunk
        chunk_overlap=200  # Overlap to prevent losing context between chunks
    )
    chunks = text_splitter.split_documents(docs)

    # Prevent ingestion an empty chunks
    chunks = [doc for doc in chunks if doc.page_content.strip()]

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
        client=persistent_client,
        collection_name=collection_input,
    )

    # Check if file is already indexed in Chroma database
    resp = is_file_already_indexed(vector_store, pdf_file)
    print('[ File already indexed in vector store? ]', resp)

    if resp:
        return 'File successfully ingested'
    else:
        return 'Something went wrong'


@mcp.tool()
def query_knowledge_database(query: str, collection: str):
    """
    Use this tool to perform a semantic similarity search againts the local vector database
    """

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma(
        embedding_function=embeddings,
        client=persistent_client,
        collection_name=collection
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    # Create the RAG prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, use your internal knowledge.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build and execute the retrieval-QA chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    if isinstance(response, dict):
        response_text = response.get("answer", response.get("result", str(response)))
    else:
        response_text = str(response)

    return f"search results: {response_text}"


def rag_chain_query(user_query_text, history, collection):
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
        client=persistent_client,
        collection_name=collection
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    # Multi Query Retriever
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
    )

    # Create the RAG prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, use your internal knowledge.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build and execute the retrieval-QA chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    if mq_retriever:
        rag_chain = create_retrieval_chain(multiquery_retriever, question_answer_chain)
    else:
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.stream({"input": user_query_text})

    response_text = ''

    print("\n[ Processing query ]")
    for chunk in response:
        print(chunk.get("answer", ""), end="", flush=True)
        if isinstance(chunk, dict) and "answer" in chunk:
            response_text += chunk["answer"]
            yield response_text
    print("\n")

    if voice_response:
        print("[ Converting response to audio ]")
        convert_to_audio(response_text)
    else:
        print("[ No conversion to audio ]")


def run_gradio_ui():
    """
    Gradio UI
    """

    initial_collections = get_existing_collections()
    default_selection = initial_collections[0] if initial_collections else None

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
                var_name_voice = gr.State(value="voice_response")
                var_name_mqr = gr.State(value="mq_retriever")
                checkbox_voice_resp = gr.Checkbox(label="Provide voice response", value=voice_response)
                checkbox_mqr = gr.Checkbox(label="Enable advanced search using MultiQueryRetriever", value=mq_retriever)
                checkbox_voice_resp.change(fn=toggle_global_vars, inputs=[var_name_voice, checkbox_voice_resp],
                                           outputs="")
                checkbox_mqr.change(fn=toggle_global_vars, inputs=[var_name_mqr, checkbox_mqr], outputs="")
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
                collection_input = gr.Textbox(
                    label="Collection name",
                    placeholder="e.g., finance_notes"
                )
                submit_btn_pdf = gr.Button("Upload file")
                pdf_response = gr.Textbox(
                    label="Response",
                    interactive=False
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Active collection")
                collection = gr.Dropdown(
                    choices=initial_collections,
                    value=default_selection,
                    label="Select vector database collection",
                    interactive=True
                )
                refresh_btn = gr.Button("Refresh collection list")

                gr.HTML("<hr>")
                gr.Markdown("## Indexed files")
                vec_db_response = gr.Textbox(
                    label="Response",
                    interactive=False
                )
                submit_btn_vec_db = gr.Button("Check indexed files")

            with gr.Column():
                chatb = gr.ChatInterface(
                    fn=rag_chain_query,
                    additional_inputs=[collection],
                    title="Chat",
                    description="Ask me"
                )
        none_state = gr.State(None)
        submit_btn_audio.click(fn=transcribe_audio, inputs=[audio_input], outputs=audio_output).then(fn=rag_chain_query,
                                                                                                     inputs=[
                                                                                                         audio_output,
                                                                                                         none_state,
                                                                                                         collection],
                                                                                                     outputs=audio_response)
        submit_btn_pdf.click(fn=pdf_file_ingestion, inputs=[pdf_file, collection_input], outputs=pdf_response
                             ).then(
            fn=refresh_collections_dropdown,
            inputs=None,
            outputs=collection
        )
        submit_btn_vec_db.click(fn=check_indexed_files, inputs=[collection], outputs=vec_db_response)
        refresh_btn.click(
            fn=refresh_collections_dropdown,
            inputs=None,
            outputs=collection
        )

    if gradio_public_endpoint:
        app.launch(share=True, prevent_thread_lock=True, quiet=True)
    elif mcp_server:
        app.launch(prevent_thread_lock=True, quiet=True)
    else:
        app.launch(prevent_thread_lock=True)


async def main():
    try:
        # Run MCP server
        if mcp_server:
            mcp_task = asyncio.create_task(mcp.run_async())
        else:
            mcp_task = None
        # Run gradio UI
        run_gradio_ui()

        while True:
            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[ Closing session ]")
    finally:
        if mcp_task and not mcp_task.done():
            print("[ Canceling MCP server task ]")
            mcp_task.cancel()  # Sends a CancelledError to the background task

            try:
                # Wait for the task to finish internal cleanup
                await mcp_task
            except asyncio.CancelledError:
                print("[ MCP server stopped ]")
            except Exception as e:
                print(f"[ Error during MCP server shutdown {e} ]")

        print("[ Shutdown complete ]")


if __name__ == "__main__":
    asyncio.run(main())