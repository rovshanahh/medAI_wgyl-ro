Use this full clean README section.

## Current scope

MedAIx is currently a research-use medical image routing and screening system with a backend and a frontend.

### What it currently does

- accepts an uploaded image through the web interface
- rejects many clearly irrelevant inputs before analysis
- routes supported images into one of the currently supported pipelines:
  - chest X-ray
  - bone X-ray
  - brain MRI
- runs inference for:
  - chest X-ray
  - bone X-ray
- detects brain MRI inputs, but currently stops before inference because the brain MRI route is still a placeholder in the model registry
- generates a Grad-CAM++ focus map for chest X-ray outputs
- returns a structured response including:
  - input gate result
  - detected region and modality
  - selected route and model
  - quality check result
  - OOD status
  - policy decision
  - explainability output when available

### Supported behavior right now

#### Chest X-ray
- accepted and routed to the chest X-ray model
- inference is returned
- Grad-CAM++ heatmap is generated

#### Bone X-ray
- accepted and routed to the bone X-ray model
- inference is returned

#### Brain MRI
- accepted and detected as brain MRI
- safely stopped at routing and registry level because no active brain MRI inference model is connected yet

#### Irrelevant images
- intended to be rejected before model inference through top-level gating

### What it does not do yet

- it is not a clinical system
- it is not approved for diagnosis or treatment decisions
- it does not yet run brain MRI disease inference
- it does not yet support active inference for:
  - abdomen CT
  - breast mammography
  - skin dermoscopy
  - retina fundus

### Current implementation status

- chest X-ray gate: active
- bone X-ray gate: active
- medical X-ray top-level gate: active
- brain MRI gate: active
- chest X-ray inference: active
- bone X-ray inference: active
- brain MRI inference: not yet connected

## Run the project

### Backend

From the backend folder:

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

The backend will run at:

http://localhost:8000

Frontend

From the frontend folder:

npm install
npm run dev

The frontend will run at:

http://localhost:3000

Notes
	•	make sure the backend is running before using the frontend
	•	the frontend currently sends requests to http://localhost:8000/analyze
	•	chest heatmaps are served from the backend and displayed in the frontend when available

Important

This project is for research use only and must not be used for clinical decision-making.

