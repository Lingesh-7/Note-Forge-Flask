"""
graph/nodes.py — all LangGraph node functions.
No module-level mutable state; every node receives all it needs via *state*.
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq
from tavily import TavilyClient

from graph.state import GraphState
from prompts import prompts
from utils.cache import get_from_cache, set_cache
from utils.helper import clean_topics


# ── LLM factory ───────────────────────────────────────────────────────────────

def _llm(api_key: str | None = None) -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=api_key,
    )


def _tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


# ── Planner ───────────────────────────────────────────────────────────────────

def planner_node(state: GraphState) -> GraphState:
    llm = _llm(state.get("api_key"))
    response = llm.invoke(
        prompts.planner_prompt.format_messages(syllabus=state["syllabus"])
    )

    topics = clean_topics(response.content)

    # Fallback 1: if clean_topics returns nothing, strip bullets/numbers from raw lines
    if not topics:
        raw_lines = [
            l.strip(" \t-•*0123456789.)")
            for l in response.content.splitlines()
            if l.strip()
        ]
        topics = [t for t in raw_lines if len(t) > 2][:40]

    # Fallback 2: use the syllabus lines themselves
    if not topics:
        topics = [
            l.strip() for l in state["syllabus"].splitlines()
            if l.strip()
        ][:40]

    if not topics:
        raise ValueError("Could not extract any topics from the syllabus.")

    return {
        **state,
        "topics":              topics,
        "current_topic_index": 0,
        "current_topic":       topics[0],
        "all_notes":           [],
        "all_questions":       [],
    }


# ── Researcher ────────────────────────────────────────────────────────────────

def researcher_node(state: GraphState) -> GraphState:
    topic       = state["current_topic"]
    has_book    = state["has_book"]
    vectorstore = state.get("vectorstore")

    cache_key = f"research::{topic}"
    cached    = get_from_cache(cache_key)
    if cached:
        return {**state, "research_content": cached}

    llm = _llm(state.get("api_key"))

    if has_book and vectorstore:
        docs = vectorstore.similarity_search(topic, k=4)
        seen: set[str] = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        context = "\n\n".join(d.page_content for d in unique_docs[:6])
    else:
        results = _tavily().search(query=topic, max_results=3)
        context = "\n\n".join(r["content"] for r in results["results"])

    context = context.replace("\n\n\n", "\n")[:1400]

    response = llm.invoke(
        prompts.researcher_prompt.format_messages(topic=topic, context=context)
    )
    result = response.content
    set_cache(cache_key, result)
    return {**state, "research_content": result}


# ── Writer ────────────────────────────────────────────────────────────────────

def writer_node(state: GraphState) -> GraphState:
    llm      = _llm(state.get("api_key"))
    feedback = state.get("critic_feedback", "")
    content  = state["research_content"]

    prev_notes = "\n\n".join(state.get("all_notes", [])[-2:])
    if prev_notes:
        content += f"\n\nContext from previous topics (do NOT repeat):\n{prev_notes}"

    content += (
        "\n\nInstruction: Provide a thorough, exam-focused explanation with "
        "conceptual clarity. Avoid repeating anything covered in previous topics."
    )
    if feedback:
        content += f"\n\nRevision feedback — apply these improvements:\n{feedback}"

    response = llm.invoke(
        prompts.writer_prompt.format_messages(
            topic=state["current_topic"],
            research_content=content,
        )
    )
    return {**state, "draft_notes": response.content}


# ── Critic ────────────────────────────────────────────────────────────────────

def critic_node(state: GraphState) -> GraphState:
    llm = _llm(state.get("api_key"))
    response = llm.invoke(
        prompts.critic_prompt.format_messages(
            topic=state["current_topic"],
            draft_notes=state["draft_notes"],
        )
    )
    output = response.content.lower().strip()
    passed = output.startswith("pass") and "fail" not in output
    retry  = state.get("retry_count", 0)
    if not passed:
        retry += 1
    return {
        **state,
        "critic_pass":     passed,
        "critic_feedback": response.content,
        "retry_count":     retry,
    }


# ── Exam agent ────────────────────────────────────────────────────────────────

def exam_agent_node(state: GraphState) -> GraphState:
    topic = state["current_topic"]
    set_cache(f"final_notes::{topic}", state["draft_notes"])
    return {
        **state,
        "all_notes": state.get("all_notes", []) + [state["draft_notes"]],
    }


# ── Final exam question generator ─────────────────────────────────────────────

def final_exam_node(state: GraphState) -> GraphState:
    syllabus_hash = hash(state.get("syllabus", ""))
    cache_key     = f"unit_questions::{syllabus_hash}"
    cached        = get_from_cache(cache_key)
    if cached:
        return {**state, "unit_questions": cached}

    notes  = state.get("all_notes",  [])
    topics = state.get("topics",     [])

    if not notes:
        return {**state, "unit_questions": "Questions unavailable — no notes were generated."}

    llm = _llm(state.get("api_key"))
    combined = ""
    for i, note in enumerate(notes):
        heading   = topics[i] if i < len(topics) else f"Topic {i + 1}"
        combined += f"## {heading}\n{note[:350]}\n\n"
    combined = combined[:2800]

    try:
        response = llm.invoke(
            prompts.exam_prompt.format_messages(
                topic="Complete Unit",
                final_notes=combined,
            )
        )
        questions = response.content
    except Exception:
        questions = "Questions could not be generated due to API limits."

    set_cache(cache_key, questions)
    return {**state, "unit_questions": questions}


# ── Next topic ────────────────────────────────────────────────────────────────

def next_topic_node(state: GraphState) -> GraphState:
    topics = state["topics"]
    i = state["current_topic_index"] + 1
    if i >= len(topics):
        raise IndexError(
            f"next_topic_node: index {i} is out of range for {len(topics)} topics."
        )
    return {
        **state,
        "current_topic_index": i,
        "current_topic":       topics[i],
        "retry_count":         0,
        "critic_feedback":     "",
        "critic_pass":         False,
    }


# ── Formatter ─────────────────────────────────────────────────────────────────

def formatter_node(state: GraphState) -> GraphState:
    lines: list[str] = []
    notes  = state.get("all_notes", [])
    topics = state.get("topics", [])
    for i, topic in enumerate(topics):
        lines.append(f"# {topic}\n")
        note = notes[i] if i < len(notes) else "(notes unavailable for this topic)"
        lines.append(note)
        lines.append("")
    lines.append("\n# Important Questions\n")
    lines.append(state.get("unit_questions", "No questions generated."))
    return {**state, "final_document": "\n".join(lines)}


# ── Routing functions ─────────────────────────────────────────────────────────

def critic_router(state: GraphState) -> str:
    if state["critic_pass"] or state["retry_count"] >= 1:
        return "exam_agent"
    return "writer"


def topic_router(state: GraphState) -> str:
    if state["current_topic_index"] + 1 < len(state["topics"]):
        return "next_topic"
    return "final_exam"