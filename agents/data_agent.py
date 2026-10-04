import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.llm_pick import pick_llm
from models.schema import RouterSchema, DataAgentSchema

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst


# ============================================================
# ROUTER LLM
# ============================================================

llm = pick_llm("low")

llm_router = llm.with_structured_output(RouterSchema)


# ============================================================
# ROUTER NODE
# ============================================================

def router_node(state: DataAgentSchema):

    message = state.messages[-1].content

    route_response = llm_router.invoke(message)

    state.route_response = route_response.answer

    return state


# ============================================================
# ETL NODE
# ============================================================

def etl_node(state: DataAgentSchema):

    message = state.messages[-1].content

    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(content=message)
            ]
        }
    )

    # Get messages from ETL agent's state
    state.messages = state.messages + response["messages"]

    return state


# ============================================================
# SQL NODE
# ============================================================

def sql_node(state: DataAgentSchema):

    message = state.messages[-1].content

    input_schema = {
        "messages": [],
        "user_question": message,
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    response = sql_analyst.invoke(input_schema)

    # Get messages from SQL agent's state
    state.messages = state.messages + response["messages"]

    return state


# ============================================================
# CREATE GRAPH
# ============================================================

data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node(
    "router_node",
    router_node
)

data_agent_graph.add_node(
    "etl_node",
    etl_node
)

data_agent_graph.add_node(
    "sql_node",
    sql_node
)


# START → ROUTER

data_agent_graph.add_edge(
    START,
    "router_node"
)


# ============================================================
# ROUTER EDGE
# ============================================================

def route_edge(state: DataAgentSchema) -> str:

    if state.route_response == "sql":
        return "sql_node"

    elif state.route_response == "etl":
        return "etl_node"

    else:
        raise ValueError(
            f"Invalid route response: {state.route_response}"
        )


data_agent_graph.add_conditional_edges(
    "router_node",
    route_edge,
    {
        "sql_node": "sql_node",
        "etl_node": "etl_node"
    }
)


# ============================================================
# END EDGES
# ============================================================

data_agent_graph.add_edge(
    "etl_node",
    END
)

data_agent_graph.add_edge(
    "sql_node",
    END
)


# ============================================================
# COMPILE
# ============================================================

data_agent = data_agent_graph.compile()


# ============================================================
# DRAW GRAPH
# ============================================================

from IPython.display import Image

img = Image(
    data_agent.get_graph().draw_mermaid_png()
)

with open(
    "data_agent_graph.png",
    "wb"
) as f:
    f.write(img.data)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    response = data_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="""
                    I want to extract the data from the API endpoint
                    'https://pokeapi.co/api/v2/pokemon'
                    and save it to the data/extract folder
                    in CSV format.
                    """
                )
            ],
            "route_response": ""
        }
    )

    print(response)