import os
import yaml
from src.agents.graph import build_orchestrator
from src.agents.state import InfrastructureState

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """
    Main entry point for the SAR Infrastructure Monitoring System.
    """
    print("Initializing Agentic SAR Infrastructure Monitoring System...")
    
    # Load configurations
    stac_config = load_config('configs/stac_config.yaml')
    # agent_prompts = load_config('configs/agent_prompts.yaml')
    
    # Initialize state
    initial_state: InfrastructureState = {
        "bbox": [103.85, 1.28, 103.86, 1.29], # Example Singapore BBox
        "date_range": "2023-01-01/2023-12-31",
        "stac_items": [],
        "footprints_path": "",
        "raw_data_dir": "data/raw",
        "interferogram_paths": [],
        "anomalies": [],
        "expert_assessment": "",
        "final_report": "",
        "error_message": ""
    }
    
    # Build and run the orchestrator
    orchestrator = build_orchestrator()
    
    print("Starting pipeline execution...")
    final_state = orchestrator.invoke(initial_state)
    
    print("\n--- Final Report ---")
    print(final_state["final_report"])

if __name__ == "__main__":
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    main()
