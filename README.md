# Autonomous SOC: Dynamic Evasion Attack Remediation

An LLM-orchestrated autonomous Security Operations Center (SOC) pipeline. This project demonstrates the limitations of static machine learning models against feature evasion (mimicry) attacks and implements an agentic Auto-ML solution using CrewAI, a lightweight local LLM, and a suite of machine learning models to dynamically heal detection pipelines.

## Demo

**[https://agenticcybersecurity.fabryperez.com](https://agenticcybersecurity.fabryperez.com)**

## Project Architecture

* **Streaming Engine:** Processes Parquet-based network telemetry.
* **Adversarial Injector:** Simulates Advanced Persistent Threats (APTs) by masking malicious system calls with benign profiles (Doppelgänger Evasion).
* **Static Baseline:** A standard Scikit-Learn Random Forest model.
* **Agentic Loop:** CrewAI agents utilizing a local Qwen-2.5 7B model to dynamically generate, tune, and deploy XGBoost specialist models upon detecting performance degradation.
* **Dashboard:** React + TypeScript frontend (Vite) for real-time visualization of baseline failure and agentic remediation.

## Quickstart

### Prerequisites
* [uv](https://github.com/astral-sh/uv)
* [Ollama](https://ollama.com/)
* [Node.js](https://nodejs.org/) (for frontend)

### 1. Download the Dataset
Because the raw network telemetry is too large for version control, you must download the master dataset manually. 

Download the YNU-IoT-2026 dataset from the Canadian Institute for Cybersecurity: 
**[https://www.unb.ca/cic/datasets/ynu-iot-2026.html](https://www.unb.ca/cic/datasets/ynu-iot-2026.html)**

Extract and format the pcap/csv files into `strace.parquet` and place it directly into the root directory of this project.

For this project I used the ARM strace data, although any of them should produce reasonably similar results. 

### 2. Environment Setup

```bash
git clone https://github.com/mperezfabry/Agentic-Dynamic-Evasion-Attack-Remediation.git
cd Agentic-Dynamic-Evasion-Attack-Remediation
uv sync
```

### 3. Local LLM Initialization

```bash
ollama run qwen2.5-coder:7b
```

### 4. Execution Sequence

Run the pipeline in the following order to observe the baseline and agentic responses:

**Train the Baseline Model**
```bash
uv run baseline_model.py
```

**Evaluate Baseline (Mimicry Attack Simulation)**
```bash
uv run eval_baseline.py
```

**Execute Agentic Pipeline**
```bash
uv run agentic_stream.py
```

### 5. Launch Frontend

```bash
cd frontend
npm install
npm run dev
```

For a production build:
```bash
cd frontend
npm install
npm run build
npm run preview
```

## Repository Structure

```text
.
├── agentic_stream.py          # Core streaming engine and agent logic
├── baseline_model.py          # Training script for the baseline model 
├── eval_baseline.py           # Adversarial stress-test for the baseline
├── logistic_baseline.py       # Logistic regression comparison model
├── agent_patch.py             # Agent-generated specialist patch utility
├── frontend/                  # React + TypeScript dashboard (Vite)
│   ├── src/                   # Application source code
│   ├── public/                # Static assets
│   ├── package.json           # Node dependencies
│   └── vite.config.ts         # Build configuration
├── pyproject.toml / uv.lock   # Python environment configuration
├── requirements.txt           # Pip-compatible dependency list
└── README.md                  # Project documentation
```

## Academic Context

* **Adversarial Evasion:** Demonstrates mimicry attacks bypassing tree-based ensembles.
* **Concept Drift:** Implements Dynamic Ensemble Selection (DES) to adapt to sudden telemetry shifts.
* **LLM-Driven Auto-ML:** Reduces Mean Time to Respond (MTTR) via autonomous code generation and model deployment.

Developed for New College of Florida Data Science M.S.