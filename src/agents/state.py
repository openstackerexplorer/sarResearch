from typing import TypedDict, List, Annotated
import operator

class InfrastructureState(TypedDict):
    """
    Defines the state object passed between nodes in the LangGraph workflow.
    """
    # Search parameters
    bbox: List[float]
    date_range: str
    
    # Data assets
    stac_items: List[dict]
    footprints_path: str
    raw_data_dir: str
    
    # Processed results
    interferogram_paths: List[str]
    anomalies: List[dict] # Detailed anomaly regions and scores
    
    # Reports
    expert_assessment: str
    final_report: str
    
    # Error handling
    error_message: str
