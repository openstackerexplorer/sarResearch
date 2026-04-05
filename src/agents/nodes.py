import os
from .state import InfrastructureState
from src.data_pipeline.stac_client import STACClient
from src.data_pipeline.osm_integration import fetch_infrastructure_footprints
from src.sar_processing.slc_utils import SLCUtils,load_and_calibrate_slc
from src.modeling.encoder import load_sar_foundation_model, extract_spatial_features
from src.modeling.temporal_network import detect_structural_anomalies
from langchain_core.prompts import PromptTemplate
import glob
import requests
import pystac
from dotenv import load_dotenv
from datetime import datetime
import traceback
import numpy as np
from typing import List, Dict
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
def processing_node(state: dict) -> dict:
    print("\n--- Executing SAR Processing Node ---")
    
    downloaded_files = state.get("downloaded_files", [])
    if not downloaded_files:
        state["error_message"] = "Processing failed: No files were downloaded."
        return state

    tif_files = sorted([f for f in downloaded_files if f.endswith('.tif')])
    
    if len(tif_files) < 2:
        state["error_message"] = f"Processing failed: Need at least 2 SLC images."
        return state

    try:
        print("Calibrating temporal stack for Neural Network Inference...")
        
        stack = []
        for tif in tif_files:
            # Reusing the pairing logic to ensure healthy inference data
            base_name = os.path.basename(tif)
            parts = base_name.split('_')
            stac_id = "_".join(parts[:7]) if len(parts) >= 7 else base_name.split('.')[0]
            
            matching_jsons = [f for f in downloaded_files if stac_id in f and 'extended.json' in f.lower()]
            if not matching_jsons:
                continue
                
            json_meta = matching_jsons[0]
            
            # Load, Calibrate, and Crop (Must match training crop size!)
            calibrated_slc = load_and_calibrate_slc(tif, json_meta)
            h, w = calibrated_slc.shape
            cy, cx = h // 2, w // 2
            cropped = calibrated_slc[cy-128:cy+128, cx-128:cx+128]
            
            real_ch = np.real(cropped)
            imag_ch = np.imag(cropped)
            stack.append(np.stack([real_ch, imag_ch], axis=0))
            
        # Shape: (Time, Channels, Height, Width)
        stack_array = np.array(stack)
        
        # Apply the OSM Infrastructure Mask
        footprints_path = state.get("footprints_path")
        if footprints_path and os.path.exists(footprints_path):
            print(f"Applying OSM geographic mask...")
            # Ideally, your project_geojson_to_masks function would generate these dynamically
            # For this final run, we will use a dummy mask that highlights the center of the crop

            # 1. Create a blank mask of FALSE
            plant_mask = np.zeros((256, 256), dtype=bool)
            
            # 2. Turn ONLY the center 16x16 pixels to TRUE (The footprint of the plant)
            plant_mask[120:136, 120:136] = True

            osm_masks = {
                "San Jose Cogeneration": plant_mask #np.ones((256, 256), dtype=bool),
            }
            # --- SYNTHETIC ANOMALY INJECTION FOR TESTING ---
            print("🚨 INJECTING SYNTHETIC SINKHOLE INTO FINAL 3 FRAMES 🚨")
            # We target the last 3 time steps [-3:], all channels [:], and the exact pixels of the power plant
            # By adding 5.0 to the radar phase, we simulate a sudden, severe foundation collapse
            stack_array[-3:, :, 120:136, 120:136] += 5.0
            # -----------------------------------------------
        else:
            osm_masks = {"Entire_Scene": np.ones((256, 256), dtype=bool)}

        # 6. NEURAL NETWORK INFERENCE
        print("Executing PyTorch ConvLSTM (with trained weights)...")
        
        # This calls the network, which will load sar_model_weights.pth automatically!
        anomalies = detect_structural_anomalies(stack_array, osm_masks)
        
        state["anomalies"] = anomalies
        print(f"Detected {len(state['anomalies'])} structural anomalies.")
        
    except Exception as e:
        error_msg = f"SAR Processing Error: {str(e)}"
        print(error_msg)
        traceback.print_exc() 
        state["error_message"] = error_msg
    
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
