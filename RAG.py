import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")
client = ChatGroq(model_name="llama3-70b-8192", api_key=api_key)

# Function to process uploaded PDF
def process_pdf(pdf_file):
    # Load documents
    loader = PyPDFLoader(pdf_file.name)
    data = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = text_splitter.split_documents(data)

    # Create embeddings and vector store
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.from_documents(text, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return retriever

# Prompt template
prompt_template = '''
You are a helpful assistant. Greet the user before answering.

{context}
{question}
'''
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Gradio QA chain and UI function
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
qa_chain = None  # Placeholder for QA chain

def initialize_chain(pdf_file):
    global qa_chain
    retriever = process_pdf(pdf_file)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=client,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        output_key='answer',
        get_chat_history=lambda h: h,
        verbose=False
    )
    return "PDF processed successfully. You can now ask questions!"

def query_system(user_query, chat_history):
    if qa_chain is None:
        return chat_history, "Please upload a PDF file first!"
    
    # Ensure chat_history is a list and append user query correctly
    if not isinstance(chat_history, list):
        chat_history = []

    # Append user query to chat history (as dictionary)
    chat_history.append({"role": "user", "content": user_query})

    # Query the system
    result = qa_chain(user_query)
    response_text = result["answer"]

    # Append assistant's response to chat history (as dictionary)
    chat_history.append({"role": "assistant", "content": response_text})

    return chat_history, chat_history

# Gradio UI
with gr.Blocks() as ui:
    gr.Markdown("### Document QA System with Conversational Memory")

    # State to store chat history
    chat_history = gr.State([])

    with gr.Row():
        with gr.Column():
            pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
            process_button = gr.Button("Process PDF")
            user_query = gr.Textbox(label="Ask a question:")
            submit_button = gr.Button("Submit")

        with gr.Column():
            chat_output = gr.Chatbot(label="Chat History", type="messages")
            status_output = gr.Textbox(label="Status", interactive=False)

    # Set button actions
    process_button.click(initialize_chain, [pdf_upload], status_output)
    submit_button.click(query_system, [user_query, chat_history], [chat_output, chat_history])

# Launch Gradio UI
ui.launch()
