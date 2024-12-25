# Document QA System with Conversational Memory
This repository contains a Python-based application that processes PDF documents and allows users to ask questions interactively. It leverages advanced language models and embeddings for accurate document retrieval and conversational capabilities.

# Features
Upload and process PDF documents.
Split text into manageable chunks for better comprehension.
Use embeddings for similarity-based document retrieval.
Interact with a conversational assistant powered by state-of-the-art language models.
Maintain conversational context using memory for seamless user interaction.
Clean and user-friendly interface built with Gradio.

# Getting Started
## 1. Prerequisites
Make sure the following are installed on your system:
Python 3.8+
Pip (Python Package Installer)

## 2. Clone the Repository
git clone https://github.com/Rag-PandasAIAgent/Retrieval-Augmented-Generation.git
cd Retrieval-Augmented-Generation

## 3. Install Dependencies
Install the required Python libraries:
pip install -r requirements.txt

## 4. Set Up Environment Variables
Create a .env file in the root of your project and add the following variable:
GROQ_API_KEY=your_api_key_here
+ Note: Replace your_api_key_here with your actual API key.
+ For reference, see the sample.env file provided in this repository.

# Usage
## 1. Run the Application
Launh Gradio interface:
python RAG.py
## 2. Interact with the System
Upload a PDF document via the interface.
Wait for the system to process the document.
Ask questions about the document, and the system will respond contextually.

# Project Structure
Retrieval-Augmented-Generation/
* LICENSE - License file (defines usage rights)
* RAG.py - Main application script
* requirements.txt - Python dependencies
* .gitignore - Files to ignore in version control
* .env - Environment variables (not included in the repository)
* sample.env - Template for environment variables
* README.md - Project documentation

# Technologies Used
LangChain: For chaining language models and retrieval-based QA.
FAISS: Vector search library for efficient document retrieval.
Gradio: Framework for building user-friendly web interfaces.
Python-dotenv: For managing environment variables.

# Contributing
Contributions are welcome! If you'd like to improve this project:
Fork the repository.
Create a new branch.
Commit your changes.
Submit a pull request.

# License
This project is licensed under the MIT License.

# Acknowledgments
Special thanks to:
LangChain Community for their document loaders and embeddings.
Gradio for their easy-to-use interface components.
FAISS for efficient vector-based search capabilities.

# Contact
For any questions or feedback, feel free to open an issue or contact me at [swarimabdussamad@gmail.com].
