from llm_rs.langchain import RustformersLLM
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template="""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
Answer:"""

prompt = PromptTemplate(input_variables=["instruction"],template=template,)

llm = RustformersLLM(model_path_or_repo_id="rustformers/mpt-7b-ggml",model_file="mpt-7b-instruct-q5_1-ggjt.bin",callbacks=[StreamingStdOutCallbackHandler()])

chain = LLMChain(llm=llm, prompt=prompt)

chain.run("Write a short post congratulating rustformers on their new release of their langchain integration.")