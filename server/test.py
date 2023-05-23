import os
from timeit import default_timer as timer
from typing import List
import sys
import torch
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import GPT4All
from langchain.schema import LLMResult
from langchain.vectorstores.chroma import Chroma

from query_data import llm_loader

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
history_enabled = llm_model_type != "gpt4all-j"
chatting = len(sys.argv) > 1 and sys.argv[1] == "chat"
## utility functions

import os
import textwrap


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print("\n\nAnswer:")
    print(wrap_text_preserve_newlines(llm_response["answer"]))
    print("\nSources:")
    for source in llm_response["source_documents"]:
        print(
            "  Page: "
            + str(source.metadata["page"])
            + " URL: "
            + str(source.metadata["url"])
        )


class MyCustomHandler(BaseCallbackHandler):
    def __init__(self):
        self.reset()

    def reset(self):
        self.texts = []

    def get_standalone_question(self) -> str:
        return self.texts[0].strip() if len(self.texts) > 0 else None

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when chain ends running."""
        print("\non_llm_end - response:")
        print(response)
        self.texts.append(response.generations[0][0].text)


start = timer()
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": device.type}
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
llm_loader = llm_loader(vectorstore, llm_model_type)
custom_handler = MyCustomHandler()
qa = llm_loader.get_chain(custom_handler)
end = timer()
print(f"Completed in {end - start:.3f}s")

# Chatbot loop
chat_history = []
print("Welcome to the PCI DSS v4 Chatbot! Type 'exit' to stop.")
queue = [
    "What's PCI DSS?",
    "Can you summarize the changes made from PCI DSS version 3.2.1 to version 4.0?",
    "tell me more on data loss prevention",
    "more on protecting cardholder data with strong cryptography during transmission over open, public networks",
    "exit"
]
while True:
    if chatting:
        query = input("Please enter your question: ")
    else:
        query = queue.pop(0)

    query = query.strip()
    if query.lower() == "exit":
        break

    print("\nQuestion: " + query)
    custom_handler.reset()

    start = timer()
    result = qa({"question": query, "chat_history": chat_history})
    end = timer()
    print(f"Completed in {end - start:.3f}s")

    process_llm_response(result)

    if len(chat_history) == 0:
        standalone_question = query
    else:
        standalone_question = custom_handler.get_standalone_question()

    if standalone_question is not None:
        print(f"Load relevant documents for standalone question: {standalone_question}")
        start = timer()
        docs = qa.retriever.get_relevant_documents(standalone_question)
        end = timer()
        print(f"Completed in {end - start:.3f}s")
        print(docs)

    if history_enabled:
        chat_history.append((query, result["answer"]))
