from typing import List, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    queries: List[str]     # written by Planner
    use_web: bool          # written by Planner
    evidence: str          # written by Researcher
    draft: str             # written by Writer
    approved: bool         # written by Critic
    feedback: str          # written by Critic
    revisions: int         # how many times the Critic has reviewed
    answer: str            # written by Finalize