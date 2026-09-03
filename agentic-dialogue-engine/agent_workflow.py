import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    persona: str
    rag_context: str

model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(model=model_name, temperature=0)

def retrieve_node(state: AgentState):
    # This will later connect to Tushar's Red Component
    mock_context = "Package: Full-Stack Application, Price: $2,500+"
    mock_persona = "Corporate Enterprise"
    return {"rag_context": mock_context, "persona": mock_persona}

def generate_node(state: AgentState):
    system_prompt = f"""You are a B2B sales agent for VintushTech.
    Client Persona: {state.get('persona')}
    Packages: {state.get('rag_context')}
    Keep responses concise (1-2 sentences)."""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

agent_app = workflow.compile()