from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from app.agents.citations import invalid_citations
from app.agents.llm import get_llm, message_text
from app.agents.prompts import (
    CRITIC_PROMPT,
    PLANNER_PROMPT,
    RESEARCHER_PROMPT,
    WRITER_PROMPT,
)
from app.agents.state import AgentState
from app.tools import ALL_TOOLS

MAX_TOOL_ROUNDS = 4
MAX_REVISIONS = 2
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


# ---------- Planner ----------
class Plan(BaseModel):
    queries: List[str] = Field(description="1 to 3 keyword-rich search queries")
    use_web: bool = Field(description="True only if the question needs info outside the papers")


def planner(state: AgentState) -> dict:
    plan = get_llm().with_structured_output(Plan).invoke(
        [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=state["question"])]
    )
    return {"queries": plan.queries, "use_web": plan.use_web, "revisions": 0}


# ---------- Researcher (tool-calling loop) ----------
def researcher(state: AgentState) -> dict:
    llm = get_llm().bind_tools(ALL_TOOLS)
    queries = "\n".join(f"- {q}" for q in state["queries"])
    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                f"Suggested queries:\n{queries}\n\n"
                f"Web search allowed: {state.get('use_web', False)}"
            )
        ),
    ]

    results: List[str] = []
    for _ in range(MAX_TOOL_ROUNDS):
        ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            output = tool.invoke(call["args"]) if tool else f"Unknown tool: {call['name']}"
            results.append(output)
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    # Deduplicate passages that several queries returned
    blocks: List[str] = []
    for output in results:
        for block in output.split("\n\n"):
            if block.strip() and block not in blocks:
                blocks.append(block)
    evidence = "\n\n".join(blocks) or "No evidence was found."
    return {"evidence": evidence}


# ---------- Writer ----------
def writer(state: AgentState) -> dict:
    user = f"Question: {state['question']}\n\nEvidence:\n{state['evidence']}"
    if state.get("feedback"):
        user += (
            f"\n\nYour previous draft:\n{state['draft']}"
            f"\n\nThe Critic found these problems. Fix them:\n{state['feedback']}"
        )
    ai = get_llm().invoke(
        [SystemMessage(content=WRITER_PROMPT), HumanMessage(content=user)]
    )
    return {"draft": message_text(ai)}


# ---------- Critic ----------
class Review(BaseModel):
    approved: bool = Field(description="True only if the draft is fully supported and answers the question")
    issues: List[str] = Field(default_factory=list, description="Specific problems found")


def critic(state: AgentState) -> dict:
    review = get_llm().with_structured_output(Review).invoke(
        [
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Evidence:\n{state['evidence']}\n\n"
                    f"Draft answer:\n{state['draft']}"
                )
            ),
        ]
    )

    issues = list(review.issues)
    # Programmatic check on top of the LLM check: every cited tag must exist in the evidence
    bad = invalid_citations(state["draft"], state["evidence"])
    if bad:
        issues.append(f"These citations do not appear in the evidence: {bad}")

    approved = review.approved and not bad
    return {
        "approved": approved,
        "feedback": "\n".join(f"- {i}" for i in issues),
        "revisions": state.get("revisions", 0) + 1,
    }


def route_after_critic(state: AgentState) -> str:
    if state.get("approved") or state.get("revisions", 0) > MAX_REVISIONS:
        return "finalize"
    return "writer"


# ---------- Finalize ----------
def finalize(state: AgentState) -> dict:
    answer = state["draft"]
    if not state.get("approved"):
        answer += "\n\n(Note: this answer could not be fully verified against the sources.)"
    return {"answer": answer}