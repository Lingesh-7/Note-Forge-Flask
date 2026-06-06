from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (
    planner_node,
    researcher_node,
    writer_node,
    critic_node,
    exam_agent_node,
    next_topic_node,
    final_exam_node,
    formatter_node,
    critic_router,
    topic_router,
)


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("planner",    planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer",     writer_node)
    builder.add_node("critic",     critic_node)
    builder.add_node("exam_agent", exam_agent_node)
    builder.add_node("next_topic", next_topic_node)
    builder.add_node("final_exam", final_exam_node)
    builder.add_node("formatter",  formatter_node)

    builder.set_entry_point("planner")

    builder.add_edge("planner",    "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer",     "critic")

    builder.add_conditional_edges(
        "critic",
        critic_router,
        {"writer": "writer", "exam_agent": "exam_agent"},
    )

    builder.add_conditional_edges(
        "exam_agent",
        topic_router,
        {"next_topic": "next_topic", "final_exam": "final_exam"},
    )

    builder.add_edge("next_topic", "researcher")
    builder.add_edge("final_exam", "formatter")
    builder.add_edge("formatter",  END)

    return builder.compile()
