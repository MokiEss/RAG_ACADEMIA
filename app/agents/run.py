import sys

from app.agents.citations import extract_citations
from app.agents.graph import graph


def ask(question: str, verbose: bool = True) -> dict:
    final_state: dict = {}
    for update in graph.stream({"question": question}, stream_mode="updates"):
        for node, values in update.items():
            final_state.update(values)
            if verbose:
                if node == "planner":
                    print(f"[planner]    queries={values['queries']} use_web={values['use_web']}")
                elif node == "researcher":
                    n = values["evidence"].count("\n\n") + 1
                    print(f"[researcher] gathered ~{n} passages")
                elif node == "writer":
                    print("[writer]     draft written")
                elif node == "critic":
                    print(f"[critic]     approved={values['approved']}")
                    if values["feedback"]:
                        print(values["feedback"])
    return final_state


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What retriever does the RAG paper use?"
    state = ask(question)

    print("\n=== ANSWER ===")
    print(state["answer"])
    print("\n=== SOURCES ===")
    for tag in extract_citations(state["answer"]):
        print(f"- {tag}")