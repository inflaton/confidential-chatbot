# Chat with Your Own Docs - WebSocket API

This folder is an implementation of a locally hosted chatbot API specifically focused on question answering over your own docs.

Built with [LangChain](https://github.com/hwchase17/langchain/) and [FastAPI](https://fastapi.tiangolo.com/).

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## ✅ Running locally

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and update env vars

   1. specify which LLM to use (`openai`, `gpt4all-j`, `llamacpp`, `huggingface` or `mosaicml`) by setting env var `LLM_MODEL_TYPE`.
   2. if using `openai` LLM, set env var `OPENAI_API_KEY`
   3. if using `llamacpp` LLM, comment/uncomment the lines below to choose which LlamaCpp model to use

      ```
      LLAMACPP_MODEL_PATH="../../../models/wizardLM-7B.ggml.q8_0.bin"

      # LLAMACPP_MODEL_PATH="../../../models/ggml-vic13b-q5_1.bin"

      # LLAMACPP_MODEL_PATH="../../../models/GPT4All-13B-snoozy.ggml.q5_1.bin"
      ```

   4. comment/uncomment the lines below to choose what docs to chat with

      ```
      # Index for Priceless HTML files - chunk_size=512 chunk_overlap=32
      # CHROMADB_INDEX_PATH="../../data/chromadb/"

      # Index for PCI DSS v4 PDF files - chunk_size=512 chunk_overlap=32
      # CHROMADB_INDEX_PATH="../../data/pci_dss_v4/chromadb/"

      # Index for PCI DSS v4 PDF files - chunk_size=1024 chunk_overlap=64
      CHROMADB_INDEX_PATH="../../data/pci_dss_v4/chromadb/"
      ```

3. Run automated test: `make test`
4. Chat from command line: `make chat`
5. Chat from web UI
   1. Run Python backend from this folder: `make start`
   2. Update `../../.env` by adding the following line:
      ```
      NEXT_PUBLIC_DOCS_CHAT_API_URL=ws://127.0.0.1:9000/chat
      ```
   3. Run the NextJS frontend from the project root folder by running:
      ```
      cd ../../
      yarn dev
      ```

## ✅ Comparison of open source LLMs

This project has been tested with 6 open source LLMs:

1. GPT4LL-J: dnato/ggml-gpt4all-j-v1.3-groovy.bin
2. GPT4ALL: TheBloke/GPT4All-13B-snoozy-GGML
3. Vicuna: eachadea/ggml-vicuna-13b-1.1
4. WizardLM: TheBloke/wizardLM-7B-GGML
5. WizardLM: TheBloke/wizardLM-7B-HF
6. MPT-1b-RedPajama-200b-dolly: mosaicml/mpt-1b-redpajama-200b-dolly

on platform:

- OS: Windows 10 Pro, WSL2
- Memory: 32 GB
- CPU: Intel i9-9900KF
- GPU: NVIDIA GeForce RTX 2080 SUPER, CUDA Version: 11.8

Models 1 - 4 use CPU inference, while 5 & 6 use GPU inference.

Running automated test: `make test` is to simulate a user having a conversation with LLM with the following questions:

1. What's PCI DSS?
1. Can you summarize the changes made from PCI DSS version 3.2.1 to version 4.0?
1. tell me more on data loss prevention
1. more on protecting cardholder data with strong cryptography during transmission over open, public networks

The test cannot complete with the following models:

- GPT4LL-J: dnato/ggml-gpt4all-j-v1.3-groovy.bin - model crashes when answering the second question due to out of memory (CPU)
- WizardLM: TheBloke/wizardLM-7B-HF - model crashes when answering the first question due to out of memory (GPU)

Please find the detailed comparison of other 4 models (based on log files at data/pci_dss_v4_logs/) at [this blog post](#).
