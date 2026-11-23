import os
from typing import TypedDict
from groq import Groq
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

assert os.environ.get("GROQ_API_KEY"), "Please set GROQ_API_KEY in your .env file or environment variables."

model_id = next(
    m.id for m in Groq().models.list().data
    if any(k in m.id.lower() for k in ["llama", "qwen", "mixtral", "gemma"])
    and not any(b in m.id.lower() for b in ["guard", "whisper", "tts", "compound", "vision", "prompt"])
)
print("Exact model name:", model_id)
llm = ChatGroq(model=model_id, temperature=0.3)


class AssignmentState(TypedDict):
    assignment_text: str
    assignment_topic: str
    summary: str
    evaluation: str
    feedback: str


def summarize_node(state: AssignmentState) -> dict:
    """Stage 1: Summarize the submitted student assignment."""
    print("--- Stage 1: Summarizing Assignment ---")
    prompt = (
        f"Summarize the following student assignment in 3 clear bullet points:\n\n"
        f"{state['assignment_text']}"
    )
    response = llm.invoke(prompt)
    return {"summary": response.content}


def evaluate_node(state: AssignmentState) -> dict:
    """Stage 2: Evaluate the summary against the topic and marking criteria."""
    print("--- Stage 2: Evaluating Summary ---")
    prompt = (
        f"Evaluate the following assignment summary against the topic: '{state['assignment_topic']}'.\n"
        f"Marking Criteria: Topic relevance, clarity of arguments, and balanced perspective.\n"
        f"Provide a score out of 10 and a brief 2-sentence justification.\n\n"
        f"Summary:\n{state['summary']}"
    )
    response = llm.invoke(prompt)
    return {"evaluation": response.content}


def feedback_node(state: AssignmentState) -> dict:
    """Stage 3: Prepare constructive feedback based on the evaluation."""
    print("--- Stage 3: Generating Student Feedback ---")
    prompt = (
        f"Based on the evaluation below, write encouraging and constructive feedback "
        f"for the student highlighting one strength and one area for improvement:\n\n"
        f"Evaluation:\n{state['evaluation']}"
    )
    response = llm.invoke(prompt)
    return {"feedback": response.content}


workflow = StateGraph(AssignmentState)

workflow.add_node("summarize", summarize_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("generate_feedback", feedback_node)

workflow.add_edge(START, "summarize")
workflow.add_edge("summarize", "evaluate")
workflow.add_edge("evaluate", "generate_feedback")
workflow.add_edge("generate_feedback", END)

app = workflow.compile()

with open("sequential_workflow_graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

inputs = {
    "assignment_topic": "The impact of artificial intelligence on education",
    "assignment_text": (
        "Artificial intelligence is transforming modern education by enabling personalized "
        "learning experiences for students. Intelligent tutoring systems adapt to individual "
        "learning paces, while automated grading tools save teachers significant time. "
        "However, challenges such as data privacy, algorithmic bias, and over-reliance on AI "
        "tools must be addressed to ensure equitable access and academic integrity."
    )
}

result = app.invoke(inputs)

print("================ 1. ASSIGNMENT SUMMARY ================\n")
print(result["summary"])

print("\n================ 2. EVALUATION ================\n")
print(result["evaluation"])

print("\n================ 3. STUDENT FEEDBACK ================\n")
print(result["feedback"])


class RouterState(TypedDict):
    query: str
    category: str
    answer: str


def classify_query(state: RouterState):
    q = state["query"].lower()
    if "python" in q or "code" in q or "vs code" in q:
        return {"category": "technical"}
    elif "fee" in q or "bonafide" in q or "office" in q:
        return {"category": "admin"}
    else:
        return {"category": "academic"}


def academic_node(state: RouterState):
    res = llm.invoke("Give a short 2-line academic answer: " + state["query"])
    return {"answer": res.content}


def admin_node(state: RouterState):
    res = llm.invoke("Give a short 2-line college office answer: " + state["query"])
    return {"answer": res.content}


def technical_node(state: RouterState):
    res = llm.invoke("Give a short 2-line IT technical answer: " + state["query"])
    return {"answer": res.content}


def route_next(state: RouterState):
    if state["category"] == "technical":
        return "technical_node"
    elif state["category"] == "admin":
        return "admin_node"
    return "academic_node"


router_graph = StateGraph(RouterState)

router_graph.add_node("classify_query", classify_query)
router_graph.add_node("academic_node", academic_node)
router_graph.add_node("admin_node", admin_node)
router_graph.add_node("technical_node", technical_node)

router_graph.add_edge(START, "classify_query")

router_graph.add_conditional_edges(
    "classify_query",
    route_next,
    {
        "academic_node": "academic_node",
        "admin_node": "admin_node",
        "technical_node": "technical_node"
    }
)

router_graph.add_edge("academic_node", END)
router_graph.add_edge("admin_node", END)
router_graph.add_edge("technical_node", END)

routing_app = router_graph.compile()

with open("routing_workflow_graph.png", "wb") as f:
    f.write(routing_app.get_graph().draw_mermaid_png())

queries = [
    "What is Machine Learning?",
    "How to pay college exam fee?",
    "How to fix python error in vs code?"
]

for q in queries:
    res = routing_app.invoke({"query": q})
    print("Query:", q)
    print("Route ->", res["category"].upper())
    print("Answer:", res["answer"])
    print("-" * 40)
