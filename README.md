# MedAIx

A research-use medical imaging AI platform built for TED University SENG 492 Senior Project.

MedAIx accepts medical images, validates uploaded files, routes supported inputs to the correct specialist model, runs uncertainty-aware inference, generates Grad-CAM++ visual explanations, and returns a governed policy decision through a single end-to-end pipeline.

This project is for research and educational use only. Outputs are non-diagnostic and must not be used for clinical decision-making.

---

## What it does

- Accepts uploaded images through the web interface: JPEG, PNG, TIFF, and DICOM
- Validates file format, header integrity, size limits, and image readability
- Converts DICOM files into PNG-compatible image bytes for the MVP inference pipeline
- Uses a multiclass route detector to classify uploads as:
  - brain_mri
  - bone_xray
  - chest_xray
  - unknown
- Stops unsupported, unknown, or low-confidence inputs before inference
- Routes supported images to one of three active pipelines:
  - Chest X-ray — 14-label multilabel classification (DenseNet121, CheXpert)
  - Bone X-ray — Normal / Abnormal binary classification (DenseNet121, MURA)
  - Brain MRI — 4-class tumor classification: Glioma, Meningioma, No Tumor, Pituitary (ResNet18)
- Runs MC dropout inference with epistemic and aleatoric uncertainty estimation
- Generates Grad-CAM++ heatmaps for all three active routes
- Evaluates pipeline signals through a governed decision policy: ANSWER / REFUSE / ESCALATE / REQUEST_EVIDENCE / STOP
- Returns a structured response including route result, detected region and modality, selected model, quality check, OOD status, uncertainty metrics, explainability output, warnings, and policy decision

---

## Supported routes

| Route | Region | Modality | Inference | Heatmap | Status |
|-------|--------|----------|-----------|---------|--------|
| brain_mri | Brain | MRI | 4-class tumor classification | Grad-CAM++ | Active |
| bone_xray | Bone | X-ray | Normal / Abnormal | Grad-CAM++ | Active |
| chest_xray | Chest | X-ray | 14-label multilabel classification | Grad-CAM++ | Active |
| unknown | — | — | STOP before inference | — | Active safety route |
| abdomen_ct | Abdomen | CT | — | — | Placeholder |
| breast_mammography | Breast | Mammography | — | — | Placeholder |
| skin_dermoscopy | Skin | Dermoscopy | — | — | Placeholder |
| retina_fundus | Retina | Fundus | — | — | Placeholder |

---

## Evaluation status

Current implemented checks:

- Multiclass route detector supports brain_mri, bone_xray, chest_xray, and unknown
- Unknown/random images are stopped before inference
- Brain MRI, bone X-ray, chest X-ray, and DICOM test inputs run through the governed pipeline
- Grad-CAM++ is supported for DenseNet-based and ResNet-based models
- Temporary uploaded files are deleted after processing
- The frontend displays route probabilities, selected model, inference result, OOD status, policy action, and heatmap when available

Smoke test script:

    cd backend
    python3 smoke_test_pipeline.py

Expected smoke test scenarios:

- Brain MRI sample → brain_mri route → brain_mri_resnet18
- Bone X-ray sample → bone_xray route → bone_xray_standard
- Chest X-ray sample → chest_xray route → chest_xray_mvp
- Random image → unknown route → STOP before inference
- DICOM sample → converted → routed → inference

---

## Run the project

### Requirements

- Python 3.11 or later
- Node.js 18 or later
- macOS or Linux

### Backend

    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    mkdir -p logs/heatmaps temp_uploads
    uvicorn main:app --reload --port 8000

Backend runs at: http://localhost:8000

Health check: http://localhost:8000/health

### Frontend

    cd frontend
    npm install
    npm run dev

Frontend runs at: http://localhost:3000

Make sure the backend is running before using the frontend.

---

## API endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | Backend status message |
| /health | GET | Health check |
| /analyze | POST | Upload and analyze image |
| /result/{analysis_id} | GET | Retrieve stored analysis result |
| /heatmaps/{filename} | GET | Serve generated Grad-CAM++ heatmap |

---

## Train route detector

The current routing system uses a multiclass route detector, not separate binary gate priority routing.

Expected local route dataset structure:

    datasets/routing/
    ├── train/
    │   ├── brain_mri/
    │   ├── bone_xray/
    │   ├── chest_xray/
    │   └── unknown/
    ├── val/
    │   ├── brain_mri/
    │   ├── bone_xray/
    │   ├── chest_xray/
    │   └── unknown/
    └── test/
        ├── brain_mri/
        ├── bone_xray/
        ├── chest_xray/
        └── unknown/

Training scripts:

    cd backend
    python3 prepare_routing_dataset.py
    python3 prepare_unknown_routing_images.py
    python3 train_route_detector.py

The trained route detector is saved at:

    backend/reference_data/route_detector/route_detector_model.pth

---

## Specialist models

The active specialist models are:

| Model ID | Route | Dataset / Source |
|----------|-------|------------------|
| chest_xray_mvp | chest_xray | CheXpert pretrained DenseNet121 |
| bone_xray_standard | bone_xray | MURA |
| brain_mri_resnet18 | brain_mri | Kaggle Brain Tumor MRI Dataset |

Brain MRI dataset:

    https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected local raw dataset examples:

    backend/raw_datasets/
    ├── brain-tumor-mri-dataset/
    │   ├── Training/
    │   └── Testing/
    ├── chest_xray/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── MURA-v1.1/
        ├── train/
        └── valid/

Large raw datasets are not stored in the repository.

Ignored local folders include:

    backend/raw_datasets/
    backend/datasets/
    backend/test_samples/
    backend/logs/
    backend/temp_uploads/

---

## Run evaluation / smoke tests

For current full-pipeline checks:

    cd backend
    source venv/bin/activate
    python3 smoke_test_pipeline.py

Older chest-specific evaluation can still be run if the evaluation folder is prepared:

    cd backend
    source venv/bin/activate
    python3 evaluate_chest_mvp.py

Results are saved to:

    evaluation/evaluation_results.json

---

## Project structure

    medAI_wgyl-ro/
    ├── backend/
    │   ├── main.py
    │   ├── smoke_test_pipeline.py
    │   ├── medalix/
    │   │   ├── api/                   # Orchestrator, routes, pipeline state, session store
    │   │   ├── ingestion/             # Validation and DICOM conversion
    │   │   ├── preprocessing/         # Modality-aware preprocessing
    │   │   ├── detection/             # Multiclass route detector
    │   │   ├── quality/               # Image/tensor quality assessment
    │   │   ├── ood/                   # Route-safe OOD checks
    │   │   ├── registry/              # Model registry and route-to-model resolution
    │   │   ├── inference/             # Specialist model inference with MC dropout
    │   │   ├── explainability/        # Grad-CAM++ heatmap generation
    │   │   ├── policy/                # Governed decision policy
    │   │   ├── audit/                 # PHI-free logging and audit traces
    │   │   └── retention/             # Temporary upload deletion
    │   ├── reference_data/
    │   │   ├── model_registry.json
    │   │   ├── policy_thresholds.json
    │   │   └── route_detector/
    │   │       └── route_detector_model.pth
    │   └── evaluation/
    ├── frontend/
    │   └── app/
    │       └── page.tsx
    └── README.md

---

## Pipeline stages

Every image passes through the following stages in order:

1. Ingestion validation — file format, header integrity, size limits, and readability checks
2. Format preparation — DICOM files are converted into PNG-compatible image bytes
3. Route detection — multiclass route detector predicts brain_mri, bone_xray, chest_xray, or unknown
4. Safety stop — unknown or low-confidence routes stop before inference
5. Preprocessing — image resizing, grayscale/RGB handling, and normalization
6. Quality assessment — tensor validity, resolution, contrast warning signals, and corruption indicators
7. Model registry — deterministic route-to-model resolution
8. Inference — specialist model inference with MC dropout uncertainty estimation
9. OOD detection — route-safe distribution and tensor checks
10. Explainability — Grad-CAM++ heatmap generation
11. Governed decision policy — final ANSWER / REFUSE / ESCALATE / REQUEST_EVIDENCE / STOP decision
12. Audit and cleanup — PHI-free audit trace and temporary upload deletion

---

## Frontend behavior

The frontend supports:

- Image upload for .png, .jpg, .jpeg, .tif, .tiff, and .dcm
- Browser preview for normal image files
- DICOM placeholder display for .dcm files
- Route detector result display
- Route probability display
- Selected model display
- Primary finding and confidence display when inference runs
- “No inference was run” display for stopped inputs
- OOD method and reason display
- Grad-CAM++ heatmap display when available
- Policy action and safety warnings

---

## Audit and retention

The backend records PHI-free audit information in JSONL format.

Audit records include:

- analysis ID
- timestamp
- filename
- content type
- route result
- selected model
- quality result
- OOD result
- inference summary
- explainability result
- final policy action
- pipeline stages
- DICOM conversion status when applicable

Temporary uploaded files are deleted after processing. Generated heatmaps and audit logs are stored under logs/.

---

## Safety and limitations

This project is an academic MVP and has important limitations:

- It is not a diagnostic system.
- It must not be used for clinical decision-making.
- Route detection is limited to the currently trained classes.
- DICOM support is implemented through pixel extraction and image conversion, not full clinical DICOM metadata reasoning.
- OOD checking is conservative and route-safe, but not a complete medical safety guarantee.
- Model outputs are intended for research, education, and demonstration only.

---

## Team

Gular Haji-Hasanli — Software Engineering, TED University  
Rovshana Haji-Hasanli — Software Engineering, TED University  
Arya Tabiyehzad — Computer Engineering, TED University

---

## Important

This platform is intended solely for research and educational use.

Outputs are non-diagnostic and must not be used for clinical decision-making.
