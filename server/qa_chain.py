"""Create a ChatVectorDBChain for question/answering."""
import os
from typing import Optional

import torch
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, HuggingFacePipeline, LlamaCpp
from langchain.vectorstores import VectorStore
from langchain.vectorstores.base import VectorStore
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, pipeline)


class QAChain:
    llm_model_type: str
    vectorstore: VectorStore
    llm: any

    def __init__(self, vectorstore, llm_model_type):
        self.vectorstore = vectorstore
        self.llm_model_type = llm_model_type
        self.llm = None

    def init(
        self, custom_handler: Optional[BaseCallbackHandler] = None, n_threds: int = 4
    ):
        print("initializing LLM: " + self.llm_model_type)
        print(f"       n_threds: {n_threds}")
        callbacks = [StreamingStdOutCallbackHandler()]
        if custom_handler is not None:
            callbacks.append(custom_handler)

        if self.llm is None:
            if self.llm_model_type == "openai":
                self.llm = ChatOpenAI(
                    model_name="gpt-4",
                    streaming=True,
                    callbacks=callbacks,
                    verbose=True,
                    temperature=0,
                )
            elif self.llm_model_type.startswith("gpt4all"):
                MODEL_PATH = (
                    os.environ.get("GPT4ALL_J_MODEL_PATH")
                    if self.llm_model_type == "gpt4all-j"
                    else os.environ.get("GPT4ALL_MPT_MODEL_PATH")
                )
                self.llm = GPT4All(
                    model=MODEL_PATH,
                    n_ctx=4096,
                    n_threads=n_threds,
                    seed=0,
                    temp=0.1,
                    n_predict=2048,
                    backend="gptj" if self.llm_model_type == "gpt4all-j" else "llama",
                    callbacks=callbacks,
                    verbose=True,
                    streaming=True,
                    use_mlock=True,
                )
            elif self.llm_model_type == "llamacpp":
                MODEL_PATH = os.environ.get("LLAMACPP_MODEL_PATH")
                self.llm = LlamaCpp(
                    model_path=MODEL_PATH,
                    n_ctx=4096,
                    n_threads=n_threds,
                    seed=0,
                    temperature=0,
                    max_tokens=2048,
                    callbacks=callbacks,
                    verbose=True,
                    use_mlock=True,
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
            else:
                MODEL_ID = os.environ.get("OTHER_MODEL_ID")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                pipe = pipeline(
                    "text-generation",
                    model=MODEL_ID,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                    max_new_tokens=2048,
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)

        print("initialization complete")

    def get_chain(self, tracing: bool = False) -> ConversationalRetrievalChain:
        if tracing:
            tracer = LangChainTracer()
            tracer.load_default_session()

        if self.llm is None:
            self.init()

        qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            max_tokens_limit=2048,
            return_source_documents=True,
        )

        return qa

    def call(self, inputs, streaming_handler, tracing: bool = False):
        print(inputs)

        qa = self.get_chain(tracing)

        result = (
            qa(
                inputs,
                callbacks=[streaming_handler],
            )
            if streaming_handler is not None
            else qa(inputs)
        )

        return result
