from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
import LangChainInterface
from watsonxlangchain import LangChainInterface

# IBM WatsonX credentials
creds = {
    'apikey': 'ZNPRw05jup0f2SFOCFkAtfZskeCPL6fESGaYAq1EWRq7',
    'url': 'https://us-south.ml.cloud.ibm.com'
}

# Set up the LLM using IBM WatsonX
llm = LangChainInterface(
    credentials=creds,
    model='meta-llama/llama-2-70b-chat',
    params={
        'decoding_method': 'sample',
        'max_new_tokens': 200,
        'temperature': 0.5
    },
    project_id='dd58941e-28cc-4abe-9189-0bebc2f2edec'
)

# Load and index the PDF with caching
@st.cache_resource
def load_pdf():
    pdf_name = 'what is generative ai.pdf'  # fixed assignment syntax
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)  # fixed typo: nk_overlap
    ).from_loaders(loaders)
    return index

index = load_pdf()

# Build the retrieval-based question-answering chain
chain = RetrievalQA.from_chain_type(  # fixed typo: `rom_chain_type` → `from_chain_type`
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

# Streamlit UI
st.title('Medicine Chatbot')

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Accept user input
prompt = st.chat_input('Pass Your Prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    response = chain.run(prompt)

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
