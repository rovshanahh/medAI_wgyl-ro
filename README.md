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
  - retina_fundus
  - unknown
- Stops unsupported, unknown, or low-confidence inputs before inference
- Routes supported images to one of four active medical pipelines:
  - Brain MRI — 4-class tumor classification: Glioma, Meningioma, No Tumor, Pituitary
  - Bone X-ray — Normal / Abnormal binary classification
  - Chest X-ray — 14-label multilabel classification using CheXpert labels
  - Retina fundus — 5-class diabetic retinopathy severity classification
- Runs MC dropout inference with epistemic and aleatoric uncertainty estimation
- Generates Grad-CAM++ heatmaps for all active routes
- Evaluates pipeline signals through a governed decision policy: ANSWER / REFUSE / ESCALATE / REQUEST_EVIDENCE / STOP
- Returns a structured response including route result, detected region and modality, selected model, quality check, OOD status, uncertainty metrics, explainability output, warnings, policy decision, and pipeline timing data

---

## Supported routes

| Route | Region | Modality | Inference | Heatmap | Status |
|-------|--------|----------|-----------|---------|--------|
| brain_mri | Brain | MRI | 4-class tumor classification | Grad-CAM++ | Active |
| bone_xray | Bone | X-ray | Normal / Abnormal classification | Grad-CAM++ | Active |
| chest_xray | Chest | X-ray | 14-label multilabel classification | Grad-CAM++ | Active |
| retina_fundus | Retina | Fundus | 5-class diabetic retinopathy severity classification | Grad-CAM++ | Active |
| unknown | — | — | STOP before inference | — | Active safety route |
| abdomen_ct | Abdomen | CT | — | — | Placeholder |
| breast_mammography | Breast | Mammography | — | — | Placeholder |
| skin_dermoscopy | Skin | Dermoscopy | — | — | Placeholder |

---

## Active model summary

| Model ID | Route | Architecture | Dataset / Source | Status |
|----------|-------|--------------|------------------|--------|
| brain_mri_resnet18 | brain_mri | ResNet18 | Kaggle Brain Tumor MRI Dataset | Active |
| bone_xray_standard | bone_xray | DenseNet121 | MURA | Active |
| chest_xray_mvp | chest_xray | DenseNet121 | CheXpert pretrained model | Active |
| retina_fundus_resnet18 | retina_fundus | ResNet18 | APTOS 2019 Blindness Detection | Active |

---

## Evaluation status

Current implemented checks:

- Multiclass route detector supports brain_mri, bone_xray, chest_xray, retina_fundus, and unknown
- Unknown/random images are stopped before inference
- Brain MRI, bone X-ray, chest X-ray, retina fundus, and DICOM test inputs run through the governed pipeline
- Grad-CAM++ is supported for DenseNet-based and ResNet-based models
- Temporary uploaded files are deleted after processing
- The frontend displays route probabilities, model output probabilities, selected model, inference result, OOD status, policy action, and heatmap when available
- The backend exposes route/configuration metadata through `/config` and `/routes`

Smoke test script:

    cd backend
    python3 smoke_test_pipeline.py

Expected smoke test scenarios:

- Brain MRI sample → brain_mri route → brain_mri_resnet18 → inference + heatmap
- Bone X-ray sample → bone_xray route → bone_xray_standard → inference + heatmap
- Chest X-ray sample → chest_xray route → chest_xray_mvp → inference + heatmap
- Retina fundus sample → retina_fundus route → retina_fundus_resnet18 → inference + heatmap
- Random image → unknown route → STOP before inference
- DICOM sample → converted → routed → inference + heatmap

Compact evaluation script:

    cd backend
    python3 evaluate_active_routes.py

The compact evaluation script prints a table and saves:

    evaluation/active_route_evaluation.json

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

Backend runs at:

    http://localhost:8000

Health check:

    http://localhost:8000/health

Configuration endpoint:

    http://localhost:8000/config

Routes endpoint:

    http://localhost:8000/routes

### Frontend

    cd frontend
    npm install
    npm run dev

Frontend runs at:

    http://localhost:3000

Optional frontend environment file:

    frontend/.env.local

Example:

    NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

Make sure the backend is running before using the frontend.

---

## API endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | Backend status message |
| /health | GET | Health check |
| /config | GET | Supported uploads, active routes, safety routes, disclaimer |
| /routes | GET | Active routes, safety route, inactive placeholders |
| /model-cache | GET | View cached loaded models |
| /model-cache/clear | POST | Clear cached loaded models |
| /analyze | POST | Upload and analyze image |
| /result/{analysis_id} | GET | Retrieve stored analysis result |
| /heatmaps/{filename} | GET | Serve generated Grad-CAM++ heatmap |

---

## Train route detector

The current routing system uses a multiclass route detector, not separate binary gate priority routing.

Expected local route dataset structure:

    datasets/routing/
    ├── train/
    │   ├── bone_xray/
    │   ├── brain_mri/
    │   ├── chest_xray/
    │   ├── retina_fundus/
    │   └── unknown/
    ├── val/
    │   ├── bone_xray/
    │   ├── brain_mri/
    │   ├── chest_xray/
    │   ├── retina_fundus/
    │   └── unknown/
    └── test/
        ├── bone_xray/
        ├── brain_mri/
        ├── chest_xray/
        ├── retina_fundus/
        └── unknown/

Training scripts:

    cd backend
    python3 prepare_routing_dataset.py
    python3 prepare_unknown_routing_images.py
    python3 prepare_retina_routing_dataset.py
    python3 train_route_detector.py

The trained route detector is saved at:

    backend/reference_data/route_detector/route_detector_model.pth

---

## Train specialist models

Model checkpoint files may be large. Some are generated locally from datasets, while the chest X-ray model is loaded from Hugging Face.

### Brain MRI model

Dataset:

    https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected local path:

    backend/raw_datasets/brain-tumor-mri-dataset/

Train:

    cd backend
    python3 train_brain_mri.py

Model path:

    backend/models/brain/brain_mri_resnet18.pth

### Retina fundus model

Dataset:

    APTOS 2019 Blindness Detection

Expected local path:

    backend/raw_datasets/aptos2019/

Expected files/folders:

    raw_datasets/aptos2019/
    ├── train_images/
    ├── val_images/
    ├── test_images/
    ├── train_1.csv
    ├── valid.csv
    └── test.csv

Train:

    cd backend
    python3 train_retina_fundus.py

Model path:

    backend/reference_data/models/retina_fundus_resnet18.pth

### Route detector

Train:

    cd backend
    python3 train_route_detector.py

Model path:

    backend/reference_data/route_detector/route_detector_model.pth

---

## Specialist models

The active specialist models are:

| Model ID | Route | Dataset / Source |
|----------|-------|------------------|
| chest_xray_mvp | chest_xray | CheXpert pretrained DenseNet121 |
| bone_xray_standard | bone_xray | MURA |
| brain_mri_resnet18 | brain_mri | Kaggle Brain Tumor MRI Dataset |
| retina_fundus_resnet18 | retina_fundus | APTOS 2019 Blindness Detection |

Expected local raw dataset examples:

    backend/raw_datasets/
    ├── aptos2019/
    │   ├── train_images/
    │   ├── val_images/
    │   ├── test_images/
    │   ├── train_1.csv
    │   ├── valid.csv
    │   └── test.csv
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

Large raw datasets are not stored in this repository.

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

For compact active-route evaluation:

    cd backend
    source venv/bin/activate
    python3 evaluate_active_routes.py

The active route evaluation report is saved to:

    evaluation/active_route_evaluation.json

Older chest-specific evaluation can still be run if the evaluation folder is prepared:

    cd backend
    source venv/bin/activate
    python3 evaluate_chest_mvp.py

Chest-specific results are saved to:

    evaluation/evaluation_results.json

---

## Project structure

    medAI_wgyl-ro/
    ├── backend/
    │   ├── main.py
    │   ├── smoke_test_pipeline.py
    │   ├── evaluate_active_routes.py
    │   ├── train_route_detector.py
    │   ├── train_retina_fundus.py
    │   ├── prepare_retina_routing_dataset.py
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
    │   │   ├── route_detector/
    │   │   │   └── route_detector_model.pth
    │   │   └── models/
    │   │       └── retina_fundus_resnet18.pth
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
3. Route detection — multiclass route detector predicts brain_mri, bone_xray, chest_xray, retina_fundus, or unknown
4. Safety stop — unknown or low-confidence routes stop before inference
5. Preprocessing — modality-aware resizing, grayscale/RGB handling, and normalization
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
- Route-aware review titles
- Supported-route cards loaded from backend `/config`
- Route detector result display
- Route probability display
- Model output probability display
- Selected model display
- Model output and confidence display when inference runs
- “No inference was run” display for stopped inputs
- OOD method and reason display
- Grad-CAM++ heatmap display when available
- Policy action and safety warnings
- Environment-based backend URL through `NEXT_PUBLIC_BACKEND_URL`

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
- timing summary
- DICOM conversion status when applicable

Temporary uploaded files are deleted after processing. Generated heatmaps and audit logs are stored under logs/.

---

## Safety and limitations

This project is an academic MVP and has important limitations:

- It is not a diagnostic system.
- It must not be used for clinical decision-making.
- Route detection is limited to the currently trained classes.
- DICOM support is implemented through pixel extraction and image conversion, not full clinical DICOM metadata reasoning.
- CT, mammography, dermoscopy, and other future modalities are placeholders unless explicitly activated in the registry and inference layer.
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
