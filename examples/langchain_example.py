from llm_rs.langchain import RustformersLLM
from llm_rs import KnownModels, SessionConfig
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template="""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
Answer:"""

prompt = PromptTemplate(input_variables=["instruction"],template=template,)

llm = RustformersLLM(model_path_or_repo_id="TheBloke/Nous-Hermes-13B-GGML",
                     model_file="nous-hermes-13b.ggmlv3.q4_K_S.bin",
                     verbose=True,
                     model_type=KnownModels.Llama,
                     session_config=SessionConfig(use_gpu=True),
                     callbacks=[StreamingStdOutCallbackHandler()]
)

chain = LLMChain(llm=llm, prompt=prompt)

chain.run("Write me some Cypher Querry language examples for Neo4j. Try to use the example movie dataset. Give me 5 examples of how to create nodes and relationships and how to query them.")