from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from agents.speaker_labeler import speaker_labeler_agent
from agents.deal_intel import deal_intel_agent
from agents.objection_mapper import objection_mapper_agent
from agents.proposal_crafter import proposal_crafter_agent

class AgentState(TypedDict):
    raw_transcript: str
    structured_transcript: str
    deal_intel: str
    objections: str
    proposal: str
    final_summary: str

def speaker_labeler_node(state: AgentState):
    result = speaker_labeler_agent(state["raw_transcript"])
    return {"structured_transcript": result}

def deal_intel_node(state: AgentState):
    result = deal_intel_agent(state["structured_transcript"])
    return {"deal_intel": result}

def objection_node(state: AgentState):
    result = objection_mapper_agent(state["structured_transcript"])
    return {"objections": result}

def proposal_node(state: AgentState):
    result = proposal_crafter_agent(
        state["structured_transcript"], 
        state["deal_intel"], 
        state["objections"]
    )
    return {"proposal": result}

def summarize_node(state: AgentState):
    summary = f"""Deal Intelligence:\n{state['deal_intel']}\n\nObjections:\n{state['objections']}\n\nProposal:\n{state['proposal']}"""
    return {"final_summary": summary}

workflow = StateGraph(AgentState)
workflow.add_node("speaker_labeler", speaker_labeler_node)
workflow.add_node("deal_intel", deal_intel_node)
workflow.add_node("objection_mapper", objection_node)
workflow.add_node("proposal_crafter", proposal_node)
workflow.add_node("summarize", summarize_node)

workflow.set_entry_point("speaker_labeler")
workflow.add_edge("speaker_labeler", "deal_intel")
workflow.add_edge("deal_intel", "objection_mapper")
workflow.add_edge("objection_mapper", "proposal_crafter")
workflow.add_edge("proposal_crafter", "summarize")
workflow.add_edge("summarize", END)

closingforce_graph = workflow.compile()
