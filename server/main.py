"""Main entrypoint for the app."""
import json
import logging
import os
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores import VectorStore
from langchain.vectorstores.chroma import Chroma
from lcserve import serving

from qa_chain import QAChain
from schemas import ChatResponse

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

# Constants
load_dotenv(override=True)

device_type = os.environ.get("HF_EMBEDDINGS_DEVICE_TYPE") or device.type
index_path = os.environ.get("CHROMADB_INDEX_PATH")
llm_model_type = os.environ.get("LLM_MODEL_TYPE")
streaming_enabled = llm_model_type in ["openai", "llamacpp"]

start = timer()
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": device_type}
)
end = timer()

print(f"Completed in {end - start:.3f}s")

start = timer()

print(f"Load index from {index_path}")

if not os.path.isdir(index_path):
    raise ValueError(f"{index_path} does not exist!")
else:
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=index_path)

end = timer()

print(f"Completed in {end - start:.3f}s")

start = timer()
qa_chain = QAChain(vectorstore, llm_model_type)
qa_chain.init()
end = timer()
print(f"Completed in {end - start:.3f}s")


@serving(websocket=True)
def chat(question: str, history: Optional[List], **kwargs) -> str:
    # Get the `streaming_handler` from `kwargs`. This is used to stream data to the client.
    streaming_handler = kwargs.get("streaming_handler") if streaming_enabled else None
    chat_history = []
    for element in history:
        item = (element[0] or "", element[1] or "")
        chat_history.append(item)

    start = timer()
    result = qa_chain.call(
        {"question": question, "chat_history": chat_history}, streaming_handler
    )
    end = timer()
    print(f"Completed in {end - start:.3f}s")

    resp = ChatResponse(sourceDocs=result["source_documents"])

    if not streaming_enabled:
        resp.token = result["answer"]

    return json.dumps(resp.dict())
