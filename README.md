# Document-Backed Q&A Bot

Streamlit chatbot that answers questions about a PDF by retrieving relevant passages and generating a grounded response, with the retrieved chunks shown alongside every answer.

## The problem

Asking an LLM about a long technical document without retrieval means it answers from parametric memory and invents specifics. This project grounds each answer in retrieved passages and surfaces those passages in the UI, so a user can verify the answer against the source rather than trusting it.

## How it works

PDFs are loaded from a local `docs/` folder with `PyPDFLoader` and split by `RecursiveCharacterTextSplitter` at 1000 characters with 10 characters of overlap. Chunks are embedded with HuggingFace `all-MiniLM-L12-v2` and indexed through LangChain's `VectorstoreIndexCreator`. Queries run through a `RetrievalQA` chain (`chain_type="stuff"`, `return_source_documents=True`) against Groq's `llama-3.1-8b-instant`, capped at 200 output tokens.

The index is built once per session with `@st.cache_resource` rather than on every query. Each answer renders with an expandable "Retrieved PDF Chunks" panel showing exactly what the model was given.

## Setup

```bash
pip install -r requirements.txt
```

No PDF is committed to this repository — an earlier commit contained copyrighted reference material and it has been removed. Create a `docs/` folder and add your own PDF before running:

```bash
mkdir docs
```

Set `GROQ_API_KEY` either as an environment variable or in Streamlit secrets, then:

```bash
streamlit run app.py
```

The app stops with an error if `docs/` contains no PDF.

## Design notes

**Groq over OpenAI.** Inference latency dominates the feel of a chat UI, and Groq's hosted Llama serves fast enough that the interface stays responsive.

**10-character chunk overlap is small.** It minimizes duplication across retrieved chunks but risks splitting a sentence across a boundary; 100–200 characters is more typical and worth testing against retrieval quality.

**200-token output cap.** Keeps answers terse and fast, at the cost of truncating longer explanations. A deliberate trade for a Q&A interface.

## Results

Not yet measured. Retrieval quality here is unevaluated — see limitations.

## Known limitations

**No retrieval evaluation.** Nothing measures whether the retrieved chunks actually contain the answer. Recall@k against a small labeled question set is the obvious next step, and would make the chunking choices above testable rather than guessed.

**Chunk size and overlap are untuned defaults.**

**No out-of-scope guardrail.** If retrieval returns nothing relevant, the model still answers from parametric memory — the exact failure this design is meant to prevent.

**Single-corpus scope.** All PDFs in `docs/` are indexed together with no per-document metadata, so multi-document retrieval cannot be filtered or attributed by source.

## Stack

Python · Streamlit · LangChain · HuggingFace all-MiniLM-L12-v2 · Groq (llama-3.1-8b-instant)

## License

MIT
