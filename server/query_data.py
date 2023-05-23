"""Create a ChatVectorDBChain for question/answering."""
import os
from typing import Optional

import torch
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, HuggingFacePipeline, LlamaCpp
from langchain.vectorstores import VectorStore
from langchain.vectorstores.base import VectorStore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)


class LLMLoader:
    llm_model_type: str
    vectorstore: VectorStore
    llm: any

    def __init__(self, vectorstore, llm_model_type):
        self.vectorstore = vectorstore
        self.llm_model_type = llm_model_type
        self.llm = None

    def init(self, stream_handler):
        print("initializing LLM: " + self.llm_model_type)
        callbacks = [StreamingStdOutCallbackHandler()]
        if stream_handler is not None:
            callbacks.append(stream_handler)

        if self.llm is None:
            if self.llm_model_type == "gpt4all-j":
                MODEL_PATH = os.environ.get("GPT4ALL_J_MODEL_PATH")
                self.llm = GPT4All(
                    model=MODEL_PATH,
                    n_ctx=2048 * 2,
                    backend="gptj",
                    callbacks=callbacks,
                    verbose=True,
                )
            elif self.llm_model_type == "llamacpp":
                MODEL_PATH = os.environ.get("LLAMACPP_MODEL_PATH")
                self.llm = LlamaCpp(
                    model_path=MODEL_PATH, n_ctx=2048, callbacks=callbacks, verbose=True
                )
            elif self.llm_model_type == "huggingface":
                MODEL_ID = os.environ.get("HUGGINGFACE_MODEL_ID")
                pipe = (
                    pipeline(
                        "text-generation",
                        model=MODEL_ID,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        max_new_tokens=2048,
                    )
                    if torch.cuda.is_available()
                    else pipeline(
                        "text-generation",
                        model=MODEL_ID,
                        device_map="auto",
                        max_new_tokens=2048,
                    )
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
            elif self.llm_model_type == "mosaicml":
                MODEL_ID = os.environ.get("MOSAICML_MODEL_ID")
                model = (
                    AutoModelForCausalLM.from_pretrained(
                        MODEL_ID,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        max_seq_len=2048,
                        verbose=True,
                    )
                    if torch.cuda.is_available()
                    else AutoModelForCausalLM.from_pretrained(
                        MODEL_ID, trust_remote_code=True, max_seq_len=2048, verbose=True
                    )
                )
                model.eval()
                device = (
                    f"cuda:{torch.cuda.current_device()}"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                model.to(device)
                print(f"Model loaded on {device}")

                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

                # mtp-7b is trained to add "<|endoftext|>" at the end of generations
                stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

                # define custom stopping criteria object
                class StopOnTokens(StoppingCriteria):
                    def __call__(
                        self,
                        input_ids: torch.LongTensor,
                        scores: torch.FloatTensor,
                        **kwargs,
                    ) -> bool:
                        for stop_id in stop_token_ids:
                            if input_ids[0][-1] == stop_id:
                                return True
                        return False

                stopping_criteria = StoppingCriteriaList([StopOnTokens()])

                pipe = pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    return_full_text=True,  # langchain expects the full text
                    task="text-generation",
                    device=device,
                    # we pass model parameters here too
                    stopping_criteria=stopping_criteria,  # without this model will ramble
                    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                    top_p=0.15,  # select from top tokens whose probability add up to 15%
                    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                    max_new_tokens=2048,  # mex number of tokens to generate in the output
                    repetition_penalty=1.1,  # without this output begins repeating
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)

        print("initialization complete")

    def get_chain(
        self, stream_handler, tracing: bool = False
    ) -> ConversationalRetrievalChain:
        """Create a ChatVectorDBChain for question/answering."""
        # Construct a ChatVectorDBChain with a streaming llm for combine docs
        # and a separate, non-streaming llm for question generation
        if tracing:
            tracer = LangChainTracer()
            tracer.load_default_session()

        if self.llm_model_type == "openai":
            callbacks = [StreamingStdOutCallbackHandler()]
            if stream_handler is not None:
                callbacks.append(stream_handler)

            self.llm = ChatOpenAI(
                model_name="gpt-4",
                streaming=True,
                callbacks=callbacks,
                verbose=True,
                temperature=0,
            )
        elif self.llm is None:
            self.init(stream_handler)

        qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            max_tokens_limit=2048,
            return_source_documents=True,
        )

        return qa
