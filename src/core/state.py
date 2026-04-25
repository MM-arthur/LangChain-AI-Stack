# AgentState definition - TypedDict schema for all nodes

from typing import Dict, Any, List, TypedDict, Annotated, Optional
import operator


class AgentState(TypedDict):
    # === Core I/O ===
    input_text: str
    transcript: str
    optimized_text: str
    response: str

    # === Routing ===
    pre_route: str
    intent: Dict[str, Any]          # question_type, execution_plan
    intent_mode: str                 # normal / mock_interview / interview_review / career_planning
    route_decision: str

    # === File processing ===
    file_path: str
    file_type: str
    ocr_result: Dict[str, Any]
    document_content: str

    # === RAG ===
    rag_result: Optional[str]
    rag_sources: Optional[List[str]]

    # === Web search ===
    web_search_result: Optional[str]
    web_sources: Optional[List[str]]

    # === Behavior analysis ===
    behavior_result: Optional[Dict[str, Any]]
    video_frame_data: Optional[str]  # base64 encoded video frame

    # === Conversation ===
    history: Annotated[List[Dict[str, str]], operator.add]
    messages: Annotated[List, operator.add]

    # === Mock interview ===
    mock_interview_mode: bool
    current_round: int
    interview_history: List[str]    # accumulated Q&A pairs

    # === Interview review ===
    review_report: Optional[str]

    # === Career planning ===
    career_plan: Optional[str]