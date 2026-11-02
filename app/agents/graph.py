from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    critic,
    finalize,
    planner,
    researcher,
    route_after_critic,
    writer,
)
from app.agents.state import AgentState


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)
    builder.add_node("critic", critic)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic", route_after_critic, {"writer": "writer", "finalize": "finalize"}
    )
    builder.add_edge("finalize", END)

    return builder.compile()


graph = build_graph()