import os

from haystack.agents import Agent, Tool
from haystack.agents.base import ToolsManager
from haystack.nodes import PromptNode, PromptTemplate
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import WebQAPipeline
from haystack.nodes.prompt import PromptModel


from llm_rs.haystack import RustformersInvocationLayer

from haystack_duckduckgo import DuckDuckGoAPI

model = PromptModel("rustformers/redpajama-7b-ggml",
                    max_length=512,
                    invocation_layer_class=RustformersInvocationLayer,
                    model_kwargs={"model_file":"RedPajama-INCITE-7B-Instruct-q5_1-ggjt.bin"})

pn = PromptNode(
    model,
    max_length=256,
    default_prompt_template="question-answering-with-document-scores",
    stop_words=["Observation:"]
)

search_engine_provider = DuckDuckGoAPI()
web_retriever = WebRetriever(api_key=None,search_engine_provider=search_engine_provider,top_search_results=3,top_k=2)

pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=pn)

few_shot_prompt = """
You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions correctly, you have access to the following tools:

Search: useful for when you need to Google questions. You should ask targeted questions, for example, Who is Anthony Dirrell's brother?

To answer questions, you'll need to go through multiple steps involving step-by-step thinking and selecting appropriate tools and their inputs; tools will respond with observations. When you are ready for a final answer, respond with the `Final Answer:`
Examples:
##
Question: Anthony Dirrell is the brother of which super middleweight title holder?
Thought: Let's think step by step. To answer this question, we first need to know who Anthony Dirrell is.
Tool: Search
Tool Input: Who is Anthony Dirrell?
Observation: Boxer
Thought: We've learned Anthony Dirrell is a Boxer. Now, we need to find out who his brother is.
Tool: Search
Tool Input: Who is Anthony Dirrell brother?
Observation: Andre Dirrell
Thought: We've learned Andre Dirrell is Anthony Dirrell's brother. Now, we need to find out what title Andre Dirrell holds.
Tool: Search
Tool Input: What is the Andre Dirrell title?
Observation: super middleweight
Thought: We've learned Andre Dirrell title is super middleweight. Now, we can answer the question.
Final Answer: Andre Dirrell
##
Question: What year was the party of the winner of the 1971 San Francisco mayoral election founded?
Thought: Let's think step by step. To answer this question, we first need to know who won the 1971 San Francisco mayoral election.
Tool: Search
Tool Input: Who won the 1971 San Francisco mayoral election?
Observation: Joseph Alioto
Thought: We've learned Joseph Alioto won the 1971 San Francisco mayoral election. Now, we need to find out what party he belongs to.
Tool: Search
Tool Input: What party does Joseph Alioto belong to?
Observation: Democratic Party
Thought: We've learned Democratic Party is the party of Joseph Alioto. Now, we need to find out when the Democratic Party was founded.
Tool: Search
Tool Input: When was the Democratic Party founded?
Observation: 1828
Thought: We've learned the Democratic Party was founded in 1828. Now, we can answer the question.
Final Answer: 1828
##
Question: {query}
Thought:
{transcript}
"""

few_shot_agent_template = PromptTemplate("Few shot prompt",few_shot_prompt)

prompt_node = PromptNode(
     model,
    max_length=512,
    stop_words=["Observation:"]
)

web_qa_tool = Tool(
    name="Search",
    pipeline_or_node=pipeline,
    description="useful for when you need to Google questions.",
    output_variable="results",
)

agent = Agent(
    prompt_node=prompt_node, prompt_template=few_shot_agent_template, tools_manager=ToolsManager([web_qa_tool])
)

hotpot_questions = [
    "What year was the father of the Princes in the Tower born?",
    "Name the movie in which the daughter of Noel Harrison plays Violet Trefusis.",
    "Where was the actress who played the niece in the Priest film born?",
    "Which author is English: John Braine or Studs Terkel?",
]

for question in hotpot_questions:
    result = agent.run(query=question,max_steps=2,params={"stream":True})
    print(f"\n{result}")
