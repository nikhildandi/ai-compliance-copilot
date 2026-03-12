# AI Compliance Copilot

AI-powered compliance and policy document analysis tool built with Python, Streamlit, LangChain, FAISS, and OpenAI.

## Overview

AI Compliance Copilot allows users to upload compliance or policy documents and ask questions about them. The system retrieves relevant sections from the document and generates answers grounded in the source material.

The application uses a Retrieval-Augmented Generation (RAG) architecture commonly used in enterprise AI systems for knowledge retrieval and document analysis.

## Features

- Upload PDF policy or compliance documents
- Ask questions about document content
- AI-generated answers with supporting evidence
- Document preview and metadata
- Interactive dashboard UI
- Runs NIST CSF Gap Analysis

## Tech Stack

Python  
Streamlit  
LangChain  
OpenAI API  
FAISS Vector Database  
PyPDF  

## How It Works

1. User uploads a PDF compliance document
2. The document text is extracted and split into chunks
3. Each chunk is converted into vector embeddings
4. Embeddings are stored in a FAISS vector database
5. When a user asks a question, relevant chunks are retrieved
6. The AI model generates an answer based on those chunks

## Demo
<img width="2880" height="1620" alt="64661C0F-A4B7-4974-ACE5-F1104ED23A77" src="https://github.com/user-attachments/assets/5d6c7985-11a6-4c62-b60d-d2ca59d3e75c" />



## Installation

Clone the repository:

```bash
git clone https://github.com/nikhildandi/ai-compliance-copilot.git
cd ai-compliance-copilot
```
