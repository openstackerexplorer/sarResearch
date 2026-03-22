from langgraph.graph import StateGraph, END
from .state import InfrastructureState
from .nodes import (
    data_retrieval_node,
    processing_node,
    assessment_node,
    reporting_node
)

def check_for_errors(state: InfrastructureState) -> str:
    """
    Checks if an error occurred in the current node. 
    Routes to the next step if healthy, or aborts to reporting if failed.
    """
    if state.get("error_message"):
        return "error_route"
    return "healthy_route"

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

    # # Define edges
    # workflow.set_entry_point("data_retrieval")
    # workflow.add_edge("data_retrieval", "processing")
    # workflow.add_edge("processing", "assessment")
    # workflow.add_edge("assessment", "reporting")
    # workflow.add_edge("reporting", END)

    # Define entry point
    workflow.set_entry_point("data_retrieval")

    # Conditional routing after Retrieval
    workflow.add_conditional_edges(
        "data_retrieval",
        check_for_errors,
        {
            "healthy_route": "processing",
            "error_route": "reporting" # Skip to the end to log the error
        }
    )

    # Conditional routing after Processing
    workflow.add_conditional_edges(
        "processing",
        check_for_errors,
        {
            "healthy_route": "assessment",
            "error_route": "reporting"
        }
    )

    # Assessment flows directly into Reporting
    workflow.add_edge("assessment", "reporting")
    
    # Reporting ends the pipeline
    workflow.add_edge("reporting", END)

    # Compile
    orchestrator = workflow.compile()
    
    return orchestrator
