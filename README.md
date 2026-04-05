# Agentic SAR Infrastructure Monitoring System

A modular Python framework for monitoring the structural integrity of heavy infrastructure using Synthetic Aperture Radar (SAR) data, powered by LangGraph agentic orchestration. 

This project simulates a fully automated pipeline capable of retrieving SAR data via STAC, mapping real-world infrastructure properties using OpenStreetMap, analyzing phase shifts via ConvLSTM, and leveraging Large Language Models (LLMs) to issue domain-expert risk assessments.

## Features

- **Automated STAC Ingestion**: Efficiently queries and downloads exact SLC temporal stacks from AWS S3 (Capella Space catalogs) utilizing `pystac` and `requests`. Validates items and avoids redundant downloads.
- **Infrastructure Context via OSMnx**: Automatically connects to OpenStreetMap to fetch geographic building footprints within a bounding box, specifically filtering for heavy infrastructure (e.g., combustion power plants, industrial warehouses).
- **Core SAR Processing**: Implements utilities to read complex SAR arrays (`rasterio`) and parse essential scale factors and Rational Polynomial Coefficients (RPCs) from Capella metadata. *(Note: Coregistration, projection to slant range, and interferogram generation are architectured but utilize mocked stubs)*.
- **Offline Model Training**: Includes a dedicated `train_model.py` routine which builds a temporal stack directly from Capella downloads, performs memory-efficient center-cropping (256x256), and offline trains the ConvLSTM as a predictive autoencoder against stable ground data using MSE. Computed weights are serialized to `sar_model_weights.pth`.
- **Spatial-Temporal Neural Network (ConvLSTM)**: Contains a PyTorch-based Convolutional LSTM `SARAnomalyDetector`. The `temporal_network.py` pipeline runs inference by dynamically loading the `sar_model_weights.pth`. **Demo Capability Note:** The raw anomaly score threshold for detection has been intentionally lowered (to `> 0.01` from `0.7`) to demonstrate neural network sensitivity functionality over subtle phase changes under the generated logs.
- **Synthetic Anomaly Injection**: The `nodes.py` processing pipeline deliberately isolates a central 16x16 power plant mask and injects a +5.0 phase multiplier "synthetic sinkhole" into the final three timeframes. This validates the entire path of temporal breakdown simulating severe infrastructure collapse.
- **LLM Assessment Node**: Employs LangChain and OpenAI `gpt-4o` (or Gemini) functioning as a Senior Civil Engineer expert. Translates raw ConvLSTM anomaly millimeters into actionable risk warnings dynamically based on target asset types (e.g., higher mechanical tolerance for Solar arrays vs zero tolerance for Combustion Plants).
- **Agentic Orchestration**: Complete state-driven lifecycle using LangGraph (`InfrastructureState`). Includes conditional edge routing that gracefully handles pipeline/network errors, bypasses dead processing paths, and formats resilient final reports.

## Project Structure

```text
sar_infra_monitor/
├── configs/                    # YAML configurations (e.g. stac_config.yaml)
├── data/                       # Local data storage (raw TIFs, JSON, geojson footprints) 
├── src/
│   ├── data_pipeline/          # stac_client.py (STAC ingestion) & osm_integration.py (OSMnx mapping)
│   ├── sar_processing/         # slc_utils.py (complex math) & geometry.py (stubbed RPC projection)
│   ├── modeling/               # encoder.py, temporal_network.py (ConvLSTM network), train_model.py (Trainer)
│   └── agents/                 # nodes.py (LangChain setups & Anomaly Mock), state.py, graph.py (LangGraph)
├── sar_model_weights.pth       # Trained PyTorch state weights
├── requirements.txt            # Dependency list
└── main.py                     # Entry point orchestrator
```

## Setup & Configuration

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Add a `.env` file at the project root with your API keys to support the LangChain assessment agents:
   ```env
   OPENAI_API_KEY=your_key_here
   # Or for Gemini fallback (update nodes.py selection):
   # GOOGLE_API_KEY=your_key_here
   ```

3. **Configure Settings**:
   Review limits in `configs/stac_config.yaml` to configure specific STAC search constraints and behavior.

## Usage

1. **Train the Model (Optional)**: If you don't have existing model weights, you can run the offline training step which will extract a temporal stack and train the network dynamically:
   ```bash
   python src/modeling/train_model.py
   ```

2. **Run Pipeline**: Run the main orchestrator to test the workflow (processes a predefined San Jose footprint against Capella Stack 40 data):
   ```bash
   python main.py
   ```

Upon execution, the terminal will trace the LangGraph node states:
1. Downloading predefined S3 data slices if missing.
2. Fetching and persisting geographic OSM infrastructure masks.
3. Applying arrays and masks passing to the PyTorch ConvLSTM inference (with a 5.0 injected phase shift simulating rapid degradation).
4. Synthesizing the final execution report using an LLM evaluator to parse the structural collapse.


## Demo 
[recoding.webm](https://github.com/user-attachments/assets/5d9e46c6-7c63-4f83-9299-04a1f2d555e5)

## Technical Notes

- **AI-Driven Engineering Parsing**: A core differentiator of this system is that it eschews static thresholds. Anomalies flagged by the PyTorch model are mapped to their OSM asset string and evaluated by the LLM node. It uses engineering principles to understand *what* the building operates as, allowing 15mm tolerance for flexible solar arrays but categorizing 5mm shifts as CRITICAL for high-pressure power generation structures.
- **Design Modularity**: The dictionary state `InfrastructureState` passed through the nodes decouples the ML inferences from the agentic flow. Swapping out the PyTorch foundation model or STAC sources won't impact or break the graph orchestration.
