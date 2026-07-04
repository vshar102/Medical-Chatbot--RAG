import glob
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_groq import ChatGroq
import logging
import os

# Streamlit secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    max_tokens=200,
    groq_api_key=groq_api_key
)
# Load and index a PDF with caching.
# NOTE: no sample PDF is committed to this repository. A previous commit
# included copyrighted reference-book PDFs; these have been removed.
# Place your own PDF(s) in a local "docs/" folder before running the app.
@st.cache_resource
def load_pdf():
    pdf_files = glob.glob(os.path.join("docs", "*.pdf"))
    if not pdf_files:
        st.error("No PDF found in the docs/ folder. Add a PDF to docs/ before using this app.")
        st.stop()
    loaders = [PyPDFLoader(f) for f in pdf_files]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    ).from_loaders(loaders)
    return index

index = load_pdf()
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question',
    return_source_documents=True
)

st.title('Document-Backed Q&A Bot')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Accept user input
prompt = st.chat_input('Ask something based on the PDF...')
if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        # Get response from the chain
        response = chain.invoke({"question": prompt})
        answer = response["result"]
        sources = response["source_documents"]

        # Show the answer
        st.chat_message('assistant').markdown(answer)
        st.session_state.messages.append({'role': 'assistant', 'content': answer})

        # Show retrieved PDF chunks
        with st.expander("Retrieved PDF Chunks"):
            for i, doc in enumerate(sources):
                st.markdown(f"Chunk {i+1}:\n\n{doc.page_content}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error processing query: {str(e)}")
