from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

"""
# Concept: Compare LangChain's concepts of state, nodes, and edges.

    1. State (IMOW: Memory)
        Definition: A data structure that holds the current values or context of the system.
        Purpose: Tracks what the agent knows, what’s been processed, or what decisions have been made so far.
        Behavior: It is passed between nodes and can be mutated (e.g., adding an answer or a flag to indicate success).

        You can think of it like a mutable memory object that flows through the graph.

    2. Node (IMOW: Function)
        Definition: A function (sync or async) that receives the current state and outputs a new state.
        Purpose: Encapsulates a single processing step — like generating a query, retrieving context, or invoking an LLM.
        Behavior: Think of each node as a block in a flowchart — it does one job, mutates the state, and hands it off.

    3. Edge (IMOW: Control flow)
        Definition: A transition rule that determines which node runs next based on the current state.
        Purpose: Adds control flow (like if, switch, while, etc.) to the graph.
        Behavior: Edge functions read the state and return the name of the next node to invoke.

# Example
    Step 1: User asks a question.
    Step 2: Retrieve documents.
    Step 3: Generate answer using LLM.
    Step 4: If confidence is low, regenerate or fallback.
    Step 5: Return answer.

    State:{
        "question": "What is LangChain?",
        "docs": [],
        "answer": "",
        "confidence": 0.0,
        "retry_count": 0
    }

    NODES:
        | Node Name   | Description                                         |
        | ----------- | --------------------------------------------------- |
        | `retriever` | Given a question, use a vector store to get `docs`. |
        | `generator` | Use LLM to generate an answer from docs + question. |
        | `validator` | Check if answer is confident enough.                |
        | `fallback`  | Retry or return a fallback answer if needed.        |
        | `end`       | Return final output.                                |

    EDGES:
    | From        | Edge Condition             | To          |
    | ----------- | -------------------------- | ----------- |
    | `retriever` | →                          | `generator` |
    | `generator` | →                          | `validator` |
    | `validator` | if confidence ≥ 0.7        | `end`       |
    | `validator` | else if retry\_count < 2   | `fallback`  |
    | `fallback`  | → increment retry, go back | `generator` |
    | `validator` | else                       | `end`       |

    CODE:    
        builder = StateGraph()

        # Add nodes
        builder.add_node("retriever", retriever_node)
        builder.add_node("generator", generator_node)
        builder.add_node("validator", validator_node)
        builder.add_node("fallback", fallback_node)
        builder.add_node("end", end_node)

        # Add edges
        builder.set_entry_point("retriever")
        builder.add_edge("retriever", "generator")
        builder.add_edge("generator", "validator")

        builder.add_conditional_edges("validator", decide_next_step)  # function returns node name

        # Compile the graph
        graph = builder.compile()
        result = graph.invoke(initial_state)

        def decide_next_step(state: State) -> str:
            if state["confidence"] >= 0.7:
                return "end"
            elif state["retry_count"] < 2:
                return "fallback"
            else:
                return "end"

"""

class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

model = ChatOpenAI()


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# Add the chatbot node
builder.add_node("chatbot", chatbot)

# Add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# Run the graph
input = {"messages": [HumanMessage("hi!")]}
for chunk in graph.stream(input):
    print(chunk)
