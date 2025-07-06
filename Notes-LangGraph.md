# 0 Setup
```python

# exports various apis
srcs

export OPENAI_API_KEY

python -m venv .venv
source .venv/bin/activate

.venv/bin/pip install -e .
.venv/bin/pip install -U langchain langchain-openai

# To commit to my forked repo, even though my local branch is the official repo
git remote add myfork https://github.com/thomaschangsf/langchain.git
git push myfork tom-dev-1

```

## .env file VS environment variables
* .env is loaded, when code has from dotenv import load_dotenv

* Priority Order (from highest to lowest):
    - Environment variables set in the current shell/terminal session
    - Variables from .env file
    - System-wide environment variables


## Running
```python
.venv/bin/python3 ch1/py/a-llm.py
```





# CH9
### Setup
```
cd chp9/py
../../.venv/bin/pip install -e .
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma &
```


### Relationship between LangChain, Langgraph, and LangGraph CLI
* LangChain: Provides the building blocks (LLMs, prompts, tools)
* LangGraph: Provides the graph orchestration system
* LangGraph CLI: Provides deployment and management tools

### Purpose of Local Development Server
* Compiles and loads Graph
* Create Web endpoint
* Debugging and web ui: http://localchost:2024
* Allows us to test graph before deployment

* Start Local Development server
```python


# Requires python3.11; I am using python 3.11
# 1. Take the ingestion and retrieval RAG graph
# 2. Compile them into services
# 3. Start a local server
# 4. Provide debugging tools
langgraph dev --config ch9/py/langgraph.json
```


# CH10
```python

# I need these environment variabeles
echo "LANGSMITH_API_KEY: $LANGSMITH_API_KEY"
echo "LANGSMITH_TRACING_V2: $LANGSMITH_TRACING_V2"
echo "LANGSMITH_ENDPOINT: $LANGSMITH_ENDPOINT"
echo "OPENAI_API_KEY: $OPENAI_API_KEY"

# To prevent having to include the env variable in line with the python command, I updated the 3 python files {create_rag_dataset.py, rag_graph.py, and agent_evaluation_rag.py}, so it read from the .env file
from dotenv import load_dotenv
load_dotenv()

.venv/bin/python3 ch10/py/test_langsmith.py
# WITHOUT the load_deoenv(), LANGSMITH_API_KEY=$LANGSMITH_API_KEY .venv/bin/python3 ch10/py/test_langsmith.py

# Step 1: Create the evaluation dataset
 # rag_graph.py indexes the information used to answer the questions in rag_dataset 
.venv/bin/python3 ch10/py/create_rag_dataset.py 
# output to dashboard: https://smith.langchain.com/o/2d26d43b-98ae-48d3-ba89-5d4f0e052c51/datasets/3de28ab3-a907-4af6-ba3e-787546424699

# Step 2: Create agent: index the information
.venv/bin/pip install --upgrade aiohttp
.venv/bin/python ch10/py/rag_graph.py


# Step 3: Run the evaluation
LANGSMITH_API_KEY=$LANGSMITH_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python ch10/py/agent_evaluation_rag.py


# View dashboards
1. [Go to LangSmith Dashboard](https://smith.langchain.com/o/2d26d43b-98ae-48d3-ba89-5d4f0e052c51/)
2. Navigate to Datasets → "langchain-blogs-qa"
3. Navigate to Experiments → "langchain-blogs-qa-evals"
```
