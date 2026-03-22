import os
from .state import InfrastructureState
from src.data_pipeline.stac_client import STACClient
from src.data_pipeline.osm_integration import fetch_infrastructure_footprints
from src.sar_processing.slc_utils import SLCUtils
from src.modeling.encoder import load_sar_foundation_model, extract_spatial_features
from src.modeling.temporal_network import build_convlstm_model, detect_structural_anomalies
from langchain_core.prompts import PromptTemplate
import glob
import requests
import pystac
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()


# --- 2. Initialize the LLM ---
# You can easily comment/uncomment the block you want to use for the contest

# OPTION A: OpenAI (GPT-4o)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.1, # Low temperature for strict, analytical outputs
    max_tokens=1500
)

# OPTION B: Google Gemini (Gemini 1.5 Pro)
# (Often better for massive context windows if you eventually pass in huge JSON logs)
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0.1,
#     max_tokens=1500
# )

def data_retrieval_node(state: InfrastructureState) -> InfrastructureState:
    """
    Uses a pre-filtered list of STAC IDs (from the contest CSV) to download 
    the exact SAR temporal stack from the Capella AWS S3 bucket, then fetches OSM masks.
    """
    print("\n--- Executing Data Retrieval Node ---")
    os.makedirs(state["raw_data_dir"], exist_ok=True)
    
    # The official 2026 Data Fusion Contest static STAC URL
    STAC_URL = "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json"
    
    print("Loading Capella STAC Collection...")
    try:
        collection = pystac.Collection.from_file(STAC_URL)
        all_item_links = collection.get_item_links()
    except Exception as e:
        state["error_message"] = f"Failed to connect to Capella S3: {e}"
        return state
    
    if not all_item_links:
        print("No items found in the collection.")
        return
    

    SLCurls = [link for link in all_item_links if 'SLC' in link.absolute_href]
    print(f"Found {len(SLCurls)} SLC assets in the collection.")

    downloaded_paths = []
    assets_to_download = ["HH", "VV", "metadata"] # 'data' = the .tif, 'metadata' = the .json RPCs
    
    print(f"Searching catalog for {len(state['target_stac_ids'])} specific Stack IDs...")
    
    # 1. Download specific SAR files based on the CSV IDs
    for link in SLCurls:
        # Extract the STAC ID from the URL to avoid loading every single JSON file
        item_id = link.href.split('/')[-2] 
        print(f"Checking Item ID: {link} against target list... ")
        
        if item_id in state["target_stac_ids"]:
            print(f"\nMatch found! Processing Item: {item_id}")
            item = pystac.Item.from_file(link.absolute_href)
            
            for asset_key in assets_to_download:
                if asset_key in item.assets:
                    download_url = item.assets[asset_key].href
                    filename = os.path.basename(download_url.split('?')[0])
                    filepath = os.path.join(state["raw_data_dir"], f"{item.id}_{filename}")
                    
                    # Only download if we haven't already (saves huge bandwidth on restarts)
                    if not os.path.exists(filepath):
                        print(f"Downloading '{asset_key}' -> {filepath}")
                        response = requests.get(download_url, stream=True)
                        response.raise_for_status()
                        
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk: f.write(chunk)
                    else:
                        print(f"Asset '{asset_key}' already exists locally. Skipping download.")
                        
                    downloaded_paths.append(filepath)

    if not downloaded_paths:
        state["error_message"] = "No matching STAC assets could be downloaded. Check the CSV IDs."
        return state
        
    state["downloaded_files"] = downloaded_paths

    # 2. Fetch OSM infrastructure footprints (Assuming fetch_infrastructure_footprints is imported)
    print(f"\nFetching OSM infrastructure footprints for bbox: {state['bbox']}")
    footprints = fetch_infrastructure_footprints(state["bbox"])
    
    if not footprints.empty:
        footprints_path = os.path.join(state["raw_data_dir"], "footprints.geojson")
        footprints.to_file(footprints_path, driver='GeoJSON')
        state["footprints_path"] = footprints_path
        print(f"Saved OSM infrastructure masks to: {footprints_path}")
    else:
        print("Warning: No OSM infrastructure found. Masking will be bypassed.")
        state["footprints_path"] = ""

    print("--- Data Retrieval Node Complete ---\n")
    return state

# def processing_node(state: InfrastructureState) -> InfrastructureState:
#     """
#     Triggers SAR processing and neural network inference.
#     """
#     print("--- Executing Processing Node ---")
#     # 1. Load data (Placeholder)
#     # 2. Extract features using foundation model
#     encoder = load_sar_foundation_model()
#     # features = extract_spatial_features(encoder, coregistered_stack)
    
#     # 3. Detect anomalies using ConvLSTM
#     # anomaly_detector = build_convlstm_model(input_shape=(128, 16, 16))
#     # anomalies_map = detect_structural_anomalies(anomaly_detector, features)
    
#     # Update state with detected anomalies
#     state["anomalies"] = [{"id": "substation_a", "score": 0.85, "location": [1.23, 4.56]}]
    
#     return state

def processing_node(state: InfrastructureState) -> InfrastructureState:
    """
    Ingests raw SLC temporal stacks, masks them using OSM footprints, 
    and runs the spatial-temporal anomaly detection model.
    """
    print("\n--- Executing SAR Processing Node ---")
    
    # 1. Validation Check
    if not state.get("stac_items"):
        state["error_message"] = "Processing failed: No STAC items were downloaded."
        return state

    # Locate downloaded data
    raw_files = glob.glob(os.path.join(state["raw_data_dir"], "*.tif"))
    if len(raw_files) < 2:
        state["error_message"] = "Processing failed: Need at least 2 SLC images to form an interferogram."
        return state

    print(f"Found {len(raw_files)} SAR images for processing.")

    try:
        # 2. SAR Physics: Coregistration & Interferometry
        # (In a real scenario, this calls functions from your src/sar_processing/ module)
        print("Coregistering temporal stack and computing interferograms...")
        # interferograms = compute_interferogram_stack(raw_files)
        
        # Mocking the output path for the architecture
        mock_ifg_path = os.path.join(state["raw_data_dir"], "processed_ifg_01.tif")
        state["interferogram_paths"] = [mock_ifg_path]

        # 3. Apply the OSM Infrastructure Mask
        if state.get("footprints_path") and os.path.exists(state["footprints_path"]):
            print("Applying geographic mask to isolate built infrastructure...")
            # masked_ifgs = project_mask_to_slant(state["footprints_path"], interferograms)
        else:
            print("No footprint mask found. Processing entire scene (computationally heavy).")
            # masked_ifgs = interferograms

        # 4. Neural Network Inference
        print("Extracting features via Foundation Model and running ConvLSTM anomaly detection...")
        # anomalies = detect_structural_anomalies(masked_ifgs)
        
        # Mocking the detected anomalies based on our San Jose (Stack 40) context
        state["anomalies"] = [
            {
                "asset_type": "power_substation",
                "lat": 37.321, 
                "lon": -121.875,
                "subsidence_mm": -12.4,
                "confidence_score": 0.94,
                "temporal_trend": "linear_decline"
            },
            {
                "asset_type": "industrial_roof",
                "lat": 37.319, 
                "lon": -121.871,
                "subsidence_mm": 5.2,
                "confidence_score": 0.81,
                "temporal_trend": "seasonal_thermal_expansion"
            }
        ]
        
        print(f"Detected {len(state['anomalies'])} structural anomalies.")
        
    except Exception as e:
        print(f"Error during SAR processing: {str(e)}")
        state["error_message"] = f"SAR Processing Error: {str(e)}"
    
    print("--- SAR Processing Node Complete ---\n")
    return state

# def assessment_node(state: InfrastructureState) -> InfrastructureState:
#     """
#     Evaluates detected anomalies against operational baselines.
#     """
#     print("--- Executing Assessment Node ---")
#     # Logic to filter noise and prioritize genuine risks
#     anomalies = state.get("anomalies", [])
#     if any(a["score"] > 0.8 for a in anomalies):
#         state["expert_assessment"] = "High risk detected in Substation A. Displacement exceeds 5mm threshold."
#     else:
#         state["expert_assessment"] = "No critical displacements detected."
        
#     return state

def assessment_node(state: InfrastructureState) -> InfrastructureState:
    """
    Acts as a domain expert. Evaluates detected radar anomalies against 
    the specific infrastructure types identified by OpenStreetMap.
    """
    print("\n--- Executing Domain Expert Assessment Node ---")
    
    anomalies = state.get("anomalies", [])
    if not anomalies:
        state["expert_assessment"] = "No structural anomalies detected in the temporal stack. All monitored infrastructure is stable."
        print(state["expert_assessment"])
        return state

    # This prompt instructs the LLM on how to evaluate different infrastructure types
    # based on real-world civil engineering tolerances.
    assessment_prompt = PromptTemplate.from_template("""
    You are a Senior Civil and Structural Engineering Assessor evaluating InSAR (Interferometric Synthetic Aperture Radar) deformation reports.

    You will be provided with a list of anomalous surface shifts detected on specific infrastructure assets over the past year. 
    
    Evaluate the risk level (Low, Medium, High, CRITICAL) for each asset based on these standard tolerances:
    1. Combustion Power Plants / Cogeneration: Extremely sensitive. Anything > 5mm/year is HIGH RISK due to high-pressure steam pipes.
    2. Solar Photovoltaic Arrays: Flexible. Can handle up to 15mm/year of shift without total failure.
    3. Industrial Warehouses/Depots: Moderate risk. Subsidence > 10mm/year requires structural inspection of the foundation slab.

    Detected Anomalies:
    {anomalies_data}

    Format your response as a professional Engineering Assessment Report. Detail why each anomaly is classified at its specific risk level.
    """)

    # Format the anomalies payload using the OSM data and SAR results
    # (In the real code, this matches the anomalies list generated in the processing node)
    formatted_anomalies = ""
    for idx, anomaly in enumerate(anomalies):
        # We assume the processing node merged the OSM 'name'/'power' tags with the SAR shift data
        asset_type = anomaly.get('asset_type', 'Unknown Structure')
        name = anomaly.get('name', 'Unnamed Asset')
        shift = anomaly.get('subsidence_mm', 0.0)
        
        formatted_anomalies += f"Asset {idx+1}: {name} ({asset_type}) | Detected Shift: {shift} mm\n"

    print(f"Submitting {len(anomalies)} anomalies for Expert Evaluation...")
    
    # Run the LangChain LLM
    try:
        chain = assessment_prompt | llm
        response = chain.invoke({"anomalies_data": formatted_anomalies})
        
        # Save the generated report to state
        state["expert_assessment"] = response.content
        print("Expert Assessment Generated Successfully.")
        
    except Exception as e:
        error_msg = f"Failed to generate assessment: {str(e)}"
        print(error_msg)
        state["error_message"] = error_msg

    print("--- Assessment Node Complete ---\n")
    return state

def reporting_node(state: InfrastructureState) -> InfrastructureState:
    # """
    # Synthesizes findings into a final report using an LLM.
    # """
    # print("--- Executing Reporting Node ---")
    # assessment = state["expert_assessment"]
    # # In a real scenario, an LLM would format this
    # state["final_report"] = f"### Structural Integrity Report\n\n**Finding:** {assessment}\n\n**Action Item:** Immediate physical inspection of Substation A recommended."
    
    # return state
    """
    Formats the final pipeline execution into a standardized report.
    Handles both successful executions and gracefully formatted error logs.
    """
    print("\n--- Executing Reporting Node ---")
    
    # Check if we were routed here due to an error upstream
    if state.get("error_message"):
        report = f"""
    =========================================================
    ⚠️ SAR INFRASTRUCTURE MONITORING: EXECUTION FAILED ⚠️
    =========================================================
    Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
    Target BBox: {state.get('bbox', 'Unknown')}

    CRITICAL ERROR LOG:
    {state['error_message']}

    Action Required: Please check the data pipeline inputs and STAC catalog connectivity.
    =========================================================
        """
        state["final_report"] = report.strip()
        print("Error report generated.")
        return state

    # If successful, compile the executive summary
    assessment = state.get("expert_assessment", "No assessment generated.")
    downloaded_files = state.get("downloaded_files", [])
    anomalies = state.get("anomalies", [])
    
    report = f"""
    =========================================================
    ✅ SAR INFRASTRUCTURE MONITORING: EXECUTIVE SUMMARY ✅
    =========================================================
    Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
    Target Area (BBox): {state.get('bbox')}
    Total SAR Acquisitions Processed: {len(downloaded_files) // 2} (Pairs of TIF/JSON)
    Detected Structural Anomalies: {len(anomalies)}

    ---------------------------------------------------------
    AI ENGINEERING ASSESSMENT:
    {assessment}

    ---------------------------------------------------------
    DATA PROVENANCE:
    OSM Infrastructure Mask: {state.get('footprints_path', 'None applied')}
    Processing Directory: {state.get('raw_data_dir')}
    =========================================================
    """
    
    state["final_report"] = report.strip()
    print("Executive Summary Generated Successfully.")
    
    print("--- Reporting Node Complete ---\n")
    return state
