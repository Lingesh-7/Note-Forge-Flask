from typing import TypedDict, List, Any


class GraphState(TypedDict):
    syllabus: str
    has_book: bool
    vectorstore: Any
    api_key: str
    topics: List[str]
    current_topic_index: int
    current_topic: str
    research_content: str
    draft_notes: str
    critic_feedback: str
    critic_pass: bool
    retry_count: int
    all_notes: List[str]
    all_questions: List[str]
    unit_questions: str
    final_document: str
    mode: str
