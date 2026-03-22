from typing import TypedDict, List, Annotated,Any
import operator

# class InfrastructureState(TypedDict):
#     """
#     Defines the state object passed between nodes in the LangGraph workflow.
#     """
#     # Search parameters
#     bbox: List[float]
#     date_range: str
    
#     # Data assets
#     stac_items: List[dict]
#     footprints_path: str
#     raw_data_dir: str
    
#     # Processed results
#     interferogram_paths: List[str]
#     anomalies: List[dict] # Detailed anomaly regions and scores
    
#     # Reports
#     expert_assessment: str
#     final_report: str
    
#     # Error handling
#     error_message: str

class InfrastructureState(TypedDict):
    # Pipeline Inputs
    target_stac_ids: List[str]      # The precise list of STAC IDs from our CSV filter (e.g., Stack 40)
    bbox: List[float]               # Geographic bounds for OpenStreetMap footprint extraction
    raw_data_dir: str               # Directory to save the massive .tif files
    
    # Internal State updated by nodes
    downloaded_files: List[str]     # Paths to locally saved Capella .tif and .json files
    footprints_path: str            # Path to the downloaded OSM GeoJSON
    
    # Downstream Processing
    interferogram_paths: List[str]
    anomalies: List[dict]
    expert_assessment: str
    final_report: str
    error_message: str