from langgraph.graph import StateGraph, END
from .state import InfrastructureState
from .nodes import (
    data_retrieval_node,
    processing_node,
    assessment_node,
    reporting_node
)

def build_orchestrator():
    """
    Compiles the LangGraph StateGraph defining the pipeline workflow.
    """
    # Initialize the graph
    workflow = StateGraph(InfrastructureState)

    # Add nodes
    workflow.add_node("data_retrieval", data_retrieval_node)
    workflow.add_node("processing", processing_node)
    workflow.add_node("assessment", assessment_node)
    workflow.add_node("reporting", reporting_node)

    # Define edges
    workflow.set_entry_point("data_retrieval")
    workflow.add_edge("data_retrieval", "processing")
    workflow.add_edge("processing", "assessment")
    workflow.add_edge("assessment", "reporting")
    workflow.add_edge("reporting", END)

    # Compile
    orchestrator = workflow.compile()
    
    return orchestrator
