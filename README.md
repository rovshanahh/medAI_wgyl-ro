# Governed Medical Image Analysis

A research-use medical imaging AI platform built for TED University SENG 492 Senior Project.

Governed Medical Image Analysis accepts medical images, validates uploaded files, routes supported inputs to the correct specialist model, runs uncertainty-aware inference, generates Grad-CAM++ visual explanations, and returns a governed policy decision through a single end-to-end pipeline.

This project is for research and educational use only. Outputs are non-diagnostic and must not be used for clinical decision-making.

---

## What it does

- Accepts uploaded images through the web interface: JPEG, PNG, TIFF, and DICOM
- Supports both single-image review and batch review
- Validates file format, header integrity, size limits, and image readability
- Converts DICOM files into PNG-compatible image bytes for the MVP inference pipeline
- Uses a multiclass route detector to classify uploads as:
  - abdomen_ct
  - brain_mri
  - bone_xray
  - breast_mammography
  - chest_xray
  - retina_fundus
  - skin_dermoscopy
  - unknown
- Stops unsupported, unknown, or low-confidence inputs before inference
- Allows controlled manual route confirmation when automatic routing is unsafe or uncertain
- Routes supported images to active specialist pipelines:
  - Abdomen CT — 4-class kidney CT classification: Cyst, Normal, Stone, Tumor
  - Brain MRI — 4-class tumor classification: Glioma, Meningioma, No Tumor, Pituitary
  - Bone X-ray — Normal / Abnormal binary classification
  - Breast mammography — Benign / Malignant classification
  - Chest X-ray — 14-label multilabel classification using CheXpert labels
  - Retina fundus — 5-class diabetic retinopathy severity classification
  - Skin dermoscopy — 7-class lesion classification
- Runs MC dropout inference with epistemic and aleatoric uncertainty estimation
- Generates Grad-CAM++ heatmaps for active medical routes
- Evaluates pipeline signals through a governed decision policy: ANSWER / REFUSE / ESCALATE / REQUEST_EVIDENCE / STOP
- Returns a structured response including route result, detected region and modality, selected model, quality check, OOD status, uncertainty metrics, explainability output, warnings, policy decision, and pipeline timing data
- Exports review results as JSON and human-readable PDF reports

---

## Supported routes

| Route | Region | Modality | Inference | Heatmap | Status |
|---|---|---|---|---|---|
| abdomen_ct | Abdomen | CT | Cyst / Normal / Stone / Tumor classification | Grad-CAM++ | Active |
| brain_mri | Brain | MRI | Glioma / Meningioma / No Tumor / Pituitary classification | Grad-CAM++ | Active |
| bone_xray | Bone | X-ray | Normal / Abnormal classification | Grad-CAM++ | Active |
| breast_mammography | Breast | Mammography | Benign / Malignant classification | Grad-CAM++ | Active |
| chest_xray | Chest | X-ray | 14-label multilabel classification | Grad-CAM++ | Active |
| retina_fundus | Retina | Fundus | 5-class diabetic retinopathy severity classification | Grad-CAM++ | Active |
| skin_dermoscopy | Skin | Dermoscopy | 7-class lesion classification | Grad-CAM++ | Active |
| unknown | — | — | STOP before inference | — | Active safety route |

---

## Active model summary

| Model ID | Route | Architecture | Dataset / Source | Status |
|---|---|---|---|---|
| abdomen_ct_resnet18 | abdomen_ct | ResNet18 | CT Kidney Dataset: Normal, Cyst, Tumor, Stone | Active |
| brain_mri_resnet18 | brain_mri | ResNet18 | Kaggle Brain Tumor MRI Dataset | Active |
| bone_xray_standard | bone_xray | DenseNet121 | MURA | Active |
| breast_mammography_resnet18 | breast_mammography | ResNet18 | Mammography dataset | Active |
| chest_xray_mvp | chest_xray | DenseNet121 | CheXpert pretrained model | Active |
| retina_fundus_resnet18 | retina_fundus | ResNet18 | APTOS 2019 Blindness Detection | Active |
| skin_dermoscopy_resnet18 | skin_dermoscopy | ResNet18 | Skin dermoscopy dataset | Active |

---

## Evaluation status

Current implemented checks:

- Multiclass route detector supports abdomen_ct, brain_mri, bone_xray, breast_mammography, chest_xray, retina_fundus, skin_dermoscopy, and unknown
- Unknown/random images are stopped before inference
- All active medical routes run through route detection, preprocessing, model registry resolution, inference, OOD check, explainability, and governed policy
- DICOM test input is converted and passed through the chest X-ray pipeline
- Grad-CAM++ is supported for DenseNet-based and ResNet-based models
- Temporary uploaded files are deleted after processing
- The frontend displays route probabilities, conformal routing candidates, model output probabilities, selected model, inference result, OOD status, policy action, warnings, and heatmap when available
- The frontend supports single review, batch review, manual route confirmation, JSON export, and PDF report export
- The backend exposes route/configuration metadata through `/config` and `/routes`

Smoke test script:

    cd backend
    python3 smoke_test_pipeline.py

Expected smoke test scenarios:

- Abdomen CT sample → abdomen_ct route → abdomen_ct_resnet18 → inference + heatmap
- Brain MRI sample → brain_mri route → brain_mri_resnet18 → inference + heatmap
- Bone X-ray sample → bone_xray route → bone_xray_standard → inference + heatmap
- Breast mammography sample → breast_mammography route → breast_mammography_resnet18 → inference + heatmap
- Chest X-ray sample → chest_xray route → chest_xray_mvp → inference + heatmap
- Retina fundus sample → retina_fundus route → retina_fundus_resnet18 → inference + heatmap
- Skin dermoscopy sample → skin_dermoscopy route → skin_dermoscopy_resnet18 → inference + heatmap
- Random image → unknown route → STOP before inference
- DICOM sample → converted → routed → inference + heatmap

Compact evaluation script:

    cd backend
    python3 evaluate_active_routes.py

The compact evaluation script prints a route summary table and saves:

    evaluation/active_route_evaluation.json

Safety control evaluation:

    cd backend
    python3 evaluate_safety_controls.py

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
|---|---|---|
| / | GET | Backend status message |
| /health | GET | Health check |
| /config | GET | Supported uploads, active routes, safety routes, disclaimer |
| /routes | GET | Active routes and safety route metadata |
| /model-cache | GET | View cached loaded models |
| /model-cache/clear | POST | Clear cached loaded models |
| /analyze | POST | Upload and analyze one image |
| /analyze/override | POST | Re-run an image with a manually confirmed route |
| /analyze/batch | POST | Upload and analyze multiple images |
| /result/{analysis_id} | GET | Retrieve stored analysis result |
| /heatmaps/{filename} | GET | Serve generated Grad-CAM++ heatmap |

---

## Frontend behavior

The frontend supports:

- Single image upload for .png, .jpg, .jpeg, .tif, .tiff, and .dcm
- Batch image upload and batch summary cards
- Browser preview for normal image files
- DICOM placeholder display for .dcm files
- Route-aware review titles
- Supported-route cards loaded from backend `/config`
- Manual route confirmation after STOP / ESCALATE / uncertain routing
- Route detector result display
- Route probability display
- Conformal routing candidate display
- Model output probability display
- Selected model display
- Model output and confidence display when inference runs
- “No inference was run” display for stopped inputs
- OOD method and reason display
- Grad-CAM++ heatmap display when available
- Policy action, risk category, and safety warnings
- Recent review list
- JSON report download
- PDF report download
- Batch JSON report download
- Environment-based backend URL through `NEXT_PUBLIC_BACKEND_URL`

---

## Train route detector

The current routing system uses a multiclass route detector.

Expected local route dataset structure:

    datasets/routing/
    ├── train/
    │   ├── abdomen_ct/
    │   ├── bone_xray/
    │   ├── brain_mri/
    │   ├── breast_mammography/
    │   ├── chest_xray/
    │   ├── retina_fundus/
    │   ├── skin_dermoscopy/
    │   └── unknown/
    ├── val/
    │   ├── abdomen_ct/
    │   ├── bone_xray/
    │   ├── brain_mri/
    │   ├── breast_mammography/
    │   ├── chest_xray/
    │   ├── retina_fundus/
    │   ├── skin_dermoscopy/
    │   └── unknown/
    └── test/
        ├── abdomen_ct/
        ├── bone_xray/
        ├── brain_mri/
        ├── breast_mammography/
        ├── chest_xray/
        ├── retina_fundus/
        ├── skin_dermoscopy/
        └── unknown/

Training scripts:

    cd backend
    python3 prepare_routing_dataset.py
    python3 prepare_unknown_routing_images.py
    python3 prepare_retina_routing_dataset.py
    python3 prepare_abdomen_ct_routing_dataset.py
    python3 train_route_detector.py

The trained route detector is saved at:

    backend/reference_data/route_detector/route_detector_model.pth

---

## Train specialist models

Model checkpoint files may be large. Some are generated locally from datasets, while the chest X-ray model is loaded from Hugging Face.

### Abdomen CT model

Dataset:

    CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone

Expected local path:

    backend/raw_datasets/abdomen_ct/

Prepare dataset:

    cd backend
    python3 prepare_abdomen_ct_dataset.py

Train:

    cd backend
    python3 train_abdomen_ct.py

Model path:

    backend/reference_data/models/abdomen_ct_resnet18.pth

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
|---|---|---|
| abdomen_ct_resnet18 | abdomen_ct | CT Kidney Dataset: Normal, Cyst, Tumor, Stone |
| brain_mri_resnet18 | brain_mri | Kaggle Brain Tumor MRI Dataset |
| bone_xray_standard | bone_xray | MURA |
| breast_mammography_resnet18 | breast_mammography | Mammography dataset |
| chest_xray_mvp | chest_xray | CheXpert pretrained DenseNet121 |
| retina_fundus_resnet18 | retina_fundus | APTOS 2019 Blindness Detection |
| skin_dermoscopy_resnet18 | skin_dermoscopy | Skin dermoscopy dataset |

Expected local raw dataset examples:

    backend/raw_datasets/
    ├── abdomen_ct/
    ├── aptos2019/
    ├── brain-tumor-mri-dataset/
    ├── chest_xray/
    ├── MURA-v1.1/
    └── ...

Large raw datasets are not stored in this repository.

Ignored local folders include:

    backend/raw_datasets/
    backend/datasets/
    backend/test_samples/
    backend/logs/
    backend/temp_uploads/

---

## Run evaluation / smoke tests

For route detector check:

    cd backend
    source venv/bin/activate
    python3 test_route_detector.py

For full-pipeline smoke checks:

    cd backend
    source venv/bin/activate
    python3 smoke_test_pipeline.py

For compact active-route evaluation:

    cd backend
    source venv/bin/activate
    python3 evaluate_active_routes.py

For safety control evaluation:

    cd backend
    source venv/bin/activate
    python3 evaluate_safety_controls.py

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
    │   ├── evaluate_safety_controls.py
    │   ├── test_route_detector.py
    │   ├── train_route_detector.py
    │   ├── train_abdomen_ct.py
    │   ├── prepare_abdomen_ct_dataset.py
    │   ├── prepare_abdomen_ct_routing_dataset.py
    │   ├── medalix/
    │   │   ├── api/                   # Orchestrator, routes, pipeline state, session store
    │   │   ├── ingestion/             # Validation and DICOM conversion
    │   │   ├── preprocessing/         # Modality-aware preprocessing
    │   │   ├── detection/             # Multiclass route detector and conformal router
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
    │   │       ├── abdomen_ct_resnet18.pth
    │   │       ├── breast_mammography_resnet18.pth
    │   │       ├── retina_fundus_resnet18.pth
    │   │       └── skin_dermoscopy_resnet18.pth
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
3. Route detection — multiclass route detector predicts supported route or unknown
4. Safety stop — unknown or low-confidence routes stop before inference
5. Manual route confirmation — optional controlled human confirmation for uncertain routes
6. Preprocessing — modality-aware resizing, grayscale/RGB handling, and normalization
7. Quality assessment — tensor validity, resolution, contrast warning signals, and corruption indicators
8. Conformal routing — route candidate set and confirmation behavior
9. Model registry — deterministic route-to-model resolution
10. Inference — specialist model inference with MC dropout uncertainty estimation
11. OOD detection — route-safe distribution and tensor checks
12. Explainability — Grad-CAM++ heatmap generation
13. Governed decision policy — final ANSWER / REFUSE / ESCALATE / REQUEST_EVIDENCE / STOP decision
14. Audit and cleanup — PHI-free audit trace and temporary upload deletion

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
- manual override status when applicable

Temporary uploaded files are deleted after processing. Generated heatmaps and audit logs are stored under logs/.

---

## Safety and limitations

This project is an academic MVP and has important limitations:

- It is not a diagnostic system.
- It must not be used for clinical decision-making.
- Route detection is limited to the currently trained classes.
- DICOM support is implemented through pixel extraction and image conversion, not full clinical DICOM metadata reasoning.
- OOD checking is conservative and route-safe, but not a complete medical safety guarantee.
- Manual route confirmation is intended for controlled research/demo review, not casual patient use.
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
