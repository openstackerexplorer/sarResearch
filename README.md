# Agentic SAR Infrastructure Monitoring System

A modular Python framework for monitoring the structural integrity of heavy infrastructure using Synthetic Aperture Radar (SAR) data, powered by LangGraph agentic orchestration.

## Features

- **Automated STAC Ingestion**: Efficiently query and retrieve SLC data from SAR catalogs.
- **OSM Integration**: Automatically isolate infrastructure (power plants, substations) for targeted analysis.
- **Physics-Aware SAR Processing**: Handles complex phase data for interferometric analysis.
- **SAR Foundation Model**: Extracts robust spatial features using pre-trained convolutional encoders.
- **Temporal Anomaly Detection**: Utilizes Convolutional LSTM (ConvLSTM) to detect structural shifts over time.
- **Agentic Orchestration**: Uses LangGraph to manage the pipeline, incorporating expert assessment and automated reporting.

## Project Structure

```text
sar_infra_monitor/
├── configs/                    # YAML configurations and agent prompts
├── data/                       # Local data storage (raw, masks, processed)
├── src/
│   ├── data_pipeline/         # Data ingestion (STAC, OSM)
│   ├── sar_processing/        # Radar physics and geometry
│   ├── modeling/              # ML Models (Encoder, ConvLSTM)
│   └── agents/                # LangGraph state and orchestration
├── requirements.txt           # Dependency list
└── main.py                    # Entry point
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Endpoints**:
   Update `configs/stac_config.yaml` with your STAC catalog details.

## Usage

Run the main orchestrator:

```bash
python main.py
```

## API Documentation

All modules are documented using Python docstrings. Key components include:

- `src.data_pipeline.stac_client.STACClient`: Handles SAR item search and async downloads.
- `src.sar_processing.slc_utils`: Core math for complex SAR data.
- `src.modeling.temporal_network.TemporalAnomalyDetector`: Neural network for multi-temporal analysis.
- `src.agents.graph.build_orchestrator`: Defines the agentic workflow.

## Technical Notes

- **Modularity**: Every component is decoupled, allowing researchers to swap out the foundation model or the SAR coregistration logic without affecting the agentic flow.
- **Physics-First**: The pipeline maintains complex data integrity until final feature extraction, ensuring no loss of phase information.
