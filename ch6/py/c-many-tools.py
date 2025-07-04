import ast
from typing import Annotated, TypedDict, Literal

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]

embeddings = OpenAIEmbeddings()
model = ChatOpenAI(temperature=0.1)

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={"name": tool.name}) for tool in tools],
    embeddings,
).as_retriever()


class EvaluationResult(BaseModel):
    """Result of answer evaluation."""
    is_complete: bool = True
    is_accurate: bool = True
    confidence: Literal["high", "medium", "low"] = "high"
    missing_info: str = ""
    needs_refinement: bool = False
    reasoning: str = ""


class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]
    evaluation: EvaluationResult
    iteration_count: int


def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
    res = model.bind_tools(selected_tools).invoke(state["messages"])
    return {"messages": res}


def select_tools(state: State) -> State:
    query = state["messages"][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}


def evaluate_answer(state: State) -> State:
    """Evaluate the quality and completeness of the current answer."""
    # Get the last AI message (the answer)
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_message = msg
            break
    
    if not last_message:
        return {"evaluation": EvaluationResult(is_complete=False, needs_refinement=True)}
    
    # Create evaluation prompt
    evaluation_prompt = f"""
    Evaluate the following answer to the user's question:
    
    User Question: {state['messages'][0].content}
    Current Answer: {last_message.content}
    
    Please evaluate:
    1. Is the answer complete and addresses all parts of the question?
    2. Is the answer accurate based on the information provided?
    3. What is your confidence level in this answer?
    4. What information might be missing?
    5. Does this answer need refinement?
    
    Respond with a structured evaluation.
    """
    
    # Use structured output for evaluation
    structured_evaluator = model.with_structured_output(EvaluationResult)
    evaluation = structured_evaluator.invoke(evaluation_prompt)
    
    # Increment iteration count
    iteration_count = state.get("iteration_count", 0) + 1
    
    return {"evaluation": evaluation, "iteration_count": iteration_count}


def should_refine(state: State) -> str:
    """Determine if the answer needs refinement."""
    evaluation = state["evaluation"]
    iteration_count = state.get("iteration_count", 0)
    
    # Stop if we've done too many iterations (prevent infinite loops)
    if iteration_count >= 3:
        return END
    
    # Stop if evaluation says no refinement needed
    if not evaluation.needs_refinement:
        return END
    
    # Continue refinement if needed
    return "model"


def create_refinement_prompt(state: State) -> State:
    """Create a prompt asking the model to refine the answer."""
    evaluation = state["evaluation"]
    
    refinement_prompt = f"""
    The previous answer needs refinement. Here's the evaluation:
    
    Missing Information: {evaluation.missing_info}
    Reasoning: {evaluation.reasoning}
    Confidence: {evaluation.confidence}
    
    Please provide a more complete and accurate answer. Use the available tools if needed.
    """
    
    # Add the refinement prompt to messages
    new_message = HumanMessage(content=refinement_prompt)
    return {"messages": [new_message]}


# Build the enhanced graph
builder = StateGraph(State)
builder.add_node("select_tools", select_tools)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("evaluate", evaluate_answer)
builder.add_node("create_refinement", create_refinement_prompt)

# Add edges
builder.add_edge(START, "select_tools")
builder.add_edge("select_tools", "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "evaluate")
builder.add_conditional_edges("evaluate", should_refine)
builder.add_edge("create_refinement", "model")

graph = builder.compile()

# Example usage
input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ],
    "iteration_count": 0
}

print("Running enhanced agent with self-reflection...\n")
for c in graph.stream(input):
    print(f"Step: {c}\n")
    if "evaluation" in c:
        eval_result = c["evaluation"]
        print(f"Evaluation: Complete={eval_result.is_complete}, "
              f"Accurate={eval_result.is_accurate}, "
              f"Confidence={eval_result.confidence}, "
              f"Needs Refinement={eval_result.needs_refinement}\n")
