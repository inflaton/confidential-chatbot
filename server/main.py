"""Main entrypoint for the app."""
import json
import logging
import os
from pathlib import Path
from timeit import default_timer as timer

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

from query_data import llm_loader
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

index_path = os.environ.get("CHROMADB_INDEX_PATH")
llm_model_type = os.environ.get("LLM_MODEL_TYPE")
use_streaming = llm_model_type == "openai"
history_enabled = True #llm_model_type != "gpt4all-j"

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not Path(index_path).exists():
    raise ValueError(f"{index_path} does not exist!")

embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": device.type}
)

print("loading vectorstore")
vectorstore = Chroma(embedding_function=embeddings, persist_directory=index_path)
print("DONE")

llm_loader = llm_loader(vectorstore, llm_model_type)


@app.on_event("startup")
async def startup_event():
    llm_loader.init(None)


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    readyToSendToken: bool = False

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        # print("token: " + token)
        if self.readyToSendToken and token != "":
            resp = ChatResponse(token=token)
            await self.websocket.send_json(resp.dict())

    async def on_llm_end(self, response, **kwargs) -> None:
        """Run when LLM ends running."""
        if not self.readyToSendToken:
            self.readyToSendToken = True


@app.get("/")
async def get(request: Request):
    return "PyChat API is running!"


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_handler = StreamingLLMCallbackHandler(websocket) if use_streaming else None
    chat_history = []
    qa_chain = llm_loader.get_chain(stream_handler)

    while True:
        try:
            # Receive and send back the client message
            reqBody = await websocket.receive_text()
            print(reqBody)
            req = json.loads(reqBody)
            question = req["question"]

            start = timer()

            # ERROR - Async generation not implemented for this LLM
            if use_streaming:
                stream_handler.readyToSendToken = len(chat_history) == 0
                result = await qa_chain.acall(
                    {"question": question, "chat_history": chat_history}
                )
                print("")
            else:
                result = qa_chain({"question": question, "chat_history": chat_history})
                resp = ChatResponse(token=result["answer"])
                await websocket.send_json(resp.dict())

            end = timer()
            print(f"Completed in {end - start:.3f}s")

            resp = ChatResponse(sourceDocs=result["source_documents"])
            await websocket.send_json(resp.dict())

            if history_enabled:
                chat_history.append((question, result["answer"]))

        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                error="Sorry, something went wrong. Try again.",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
