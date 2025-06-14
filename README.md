# Medical-Chatbot--RAG

# 🎯 Project Overview
Developed enterprise-grade intelligent RAG chatbot transforming manual document analysis into automated knowledge extraction, enabling instant PDF information access with source transparency.

# 💼 Business Problem
Organizations struggled with time-intensive manual document review where knowledge workers spent 60-70% of time searching PDF repositories. Legal teams, researchers, and technical professionals faced productivity bottlenecks from inefficient document querying, causing delayed decisions and increased operational costs.

# 🔧 Technical Solution
Implemented comprehensive pipeline using PyPDFLoader for PDF extraction, RecursiveCharacterTextSplitter for text chunking (1000 characters, 10-character overlap), and HuggingFace all-MiniLM-L12-v2 embeddings. Integrated Groq's LLaMA-3.1-8B-Instant model with 200-token optimization. Built Streamlit interface with real-time chat, conversation history, and expandable source displays. Utilized LangChain's VectorstoreIndexCreator with FAISS vector store and advanced caching via @st.cache_resource decorator.

# 📊 Key Results
Real-time document querying with source transparency
Modular architecture for multi-PDF scaling
Advanced caching reducing response latency
Production-ready deployment with AI reasoning verification

# 💰 Business Impact
Delivered practical RAG solution for knowledge management, academic research, legal document review, and technical documentation. Enabled answer accuracy verification through source inspection, transforming document analysis workflows and significantly reducing time-to-insight for knowledge-intensive processes across industries.

# Skills: 
Streamlit · Retrieval-Augmented Generation (RAG) Systems · Data Interpretation · Data-driven Decision Making · Natural Language Processing (NLP) · Problem Solving · HuggingFace · Vector Databases · LangChain · Critical Thinking · Vector Embeddings · Python (Programming Language) · Data Pipelines · Informatica & Elasticsearch
