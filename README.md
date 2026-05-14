# Governed Medical Image Analysis

A research-use governed medical imaging AI platform built for TED University SENG 492 Senior Project.

Governed Medical Image Analysis accepts medical images, validates uploaded files, routes supported inputs to the correct specialist model, runs uncertainty-aware inference, checks out-of-distribution risk, generates Grad-CAM++ visual explanations, and returns a governed policy decision through a single end-to-end pipeline.

This project is for research and educational use only. Outputs are non-diagnostic and must not be used for clinical decision-making.

---

## Core idea

Most medical AI systems return a prediction for every input. This project focuses on a safer question:

Should this prediction be trusted enough to show?

The system can return one of five governed actions:

| Action | Meaning |
|---|---|
| ANSWER | Output is considered reliable enough to show |
| ESCALATE | Model ran, but human review is recommended |
| REFUSE | Prediction is withheld because reliability is too low |
| REQUEST_EVIDENCE | Input quality is insufficient; better evidence is needed |
| STOP | Input is unsafe, unsupported, or non-medical; inference is blocked |

---

## What it does

- Accepts uploaded images through the web interface: JPEG, PNG, TIFF, and DICOM
- Supports single-image review and batch review
- Validates file format, header integrity, size limits, and image readability
- Converts DICOM files into PNG-compatible image bytes for the MVP inference pipeline
- Uses a trained multiclass route detector to classify uploads as:
  - abdomen_ct
  - brain_mri
  - bone_xray
  - breast_mammography
  - chest_xray
  - retina_fundus
  - skin_dermoscopy
  - unknown
- Uses a trained medical-image gate to stop likely non-medical or unsupported inputs
- Uses a pre-inference safety gate before model inference
- Stops unsupported, unknown, OOD, or unsafe inputs before inference
- Allows controlled manual route confirmation when automatic routing is unsafe or uncertain
- Routes supported images to active specialist pipelines:
  - Abdomen CT — 4-class kidney CT classification: Cyst, Normal, Stone, Tumor
  - Brain MRI — 4-class tumor classification: Glioma, Meningioma, No Tumor, Pituitary
  - Bone X-ray — Normal / Abnormal binary classification
  - Breast mammography — Benign / Malignant classification
  - Chest X-ray — 14-label multilabel classification using CheXpert labels
  - Retina fundus — 5-class diabetic retinopathy severity classification
  - Skin dermoscopy — 7-class lesion classification
- Uses deep ensemble inference for most active specialist routes
- Uses MC dropout fallback when deep ensemble is not available
- Computes reliability score, disagreement score, predictive entropy, epistemic uncertainty, and aleatoric uncertainty
- Uses trained 5-step denoising diffusion OOD detection with route-specific thresholds
- Classifies OOD risk as:
  - IN_DISTRIBUTION
  - NEAR_OOD
  - HARD_OOD
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

| Model ID | Route | Architecture | Dataset / Source | Ensemble | Status |
|---|---|---|---|---:|---|
| abdomen_ct_resnet18 | abdomen_ct | ResNet18 | CT Kidney Dataset: Normal, Cyst, Tumor, Stone | 3 models | Active |
| brain_mri_resnet18 | brain_mri | ResNet18 | Kaggle Brain Tumor MRI Dataset | 3 models | Active |
| bone_xray_standard | bone_xray | DenseNet121 | MURA | 3 models | Active |
| breast_mammography_resnet18 | breast_mammography | ResNet18 | CBIS-DDSM mammography dataset | 3 models | Active |
| chest_xray_mvp | chest_xray | DenseNet121 | CheXpert pretrained model | 1 model | Active |
| retina_fundus_resnet18 | retina_fundus | ResNet18 | APTOS 2019 Blindness Detection | 3 models | Active |
| skin_dermoscopy_resnet18 | skin_dermoscopy | ResNet18 | HAM10000 skin dermoscopy dataset | 3 models | Active |

---

## Deep ensemble coverage

| Route | Deep ensemble enabled | Ensemble members | Uncertainty method |
|---|---:|---:|---|
| abdomen_ct | Yes | 3 | deep ensemble |
| brain_mri | Yes | 3 | deep ensemble |
| bone_xray | Yes | 3 | deep ensemble |
| breast_mammography | Yes | 3 | deep ensemble |
| retina_fundus | Yes | 3 | deep ensemble |
| skin_dermoscopy | Yes | 3 | deep ensemble |
| chest_xray | No | 1 | single model + MC dropout fallback |

---

## Current evaluation results

### Active route demo results

| Route | Output | Confidence | OOD Tier | Policy | Reliability | Uncertainty | Ensemble |
|---|---|---:|---|---|---:|---|---:|
| skin_dermoscopy | Melanoma | 60.2% | IN_DISTRIBUTION | ESCALATE | 58.9% | HIGH | 3 |
| retina_fundus | Severe | 64.2% | IN_DISTRIBUTION | ESCALATE | 62.8% | HIGH | 3 |
| brain_mri | Pituitary | 90.7% | IN_DISTRIBUTION | ANSWER | 90.5% | LOW | 3 |
| abdomen_ct | Tumor | 98.0% | IN_DISTRIBUTION | ANSWER | 98.0% | LOW | 3 |
| breast_mammography | Benign | 63.3% | IN_DISTRIBUTION | ESCALATE | 62.0% | HIGH | 3 |
| chest_xray | Cardiomegaly | 84.2% | IN_DISTRIBUTION | ANSWER | 99.9% | LOW | 1 |

### OOD evaluation

Controlled evaluation results:

| Evaluation group | Samples | Expected behavior | Observed behavior | Rate |
|---|---:|---|---|---:|
| Valid medical samples | 6 | Accepted / inference allowed | All accepted | 100% |
| Synthetic / non-medical OOD samples | 41 | STOP before unsafe inference | All stopped | 100% |

Current OOD summary:

    Valid acceptance rate: 100%
    OOD rejection rate: 100%

This is a controlled evaluation result and should not be interpreted as a clinical safety guarantee.

### Specialist model metrics

| Specialist model | Dataset / task | Ensemble size | Accuracy | Macro-F1 | Status |
|---|---|---:|---:|---:|---|
| Abdomen CT | CT kidney classification | 3 | 99.8% test avg | — | Strong |
| Brain MRI | tumor type classification | 3 | 96.0% val avg | 95.9% val avg | Strong |
| Bone X-ray | MURA normal/abnormal | 3 | 81.8% val avg | 81.7% val avg | Moderate |
| Retina Fundus | DR severity grading | 3 | 83.0% test avg | 64.9% test avg | Moderate, imbalanced |
| Breast Mammography | benign/malignant | 3 | 71.0% test avg | — | Weak/conservative |
| Skin Dermoscopy | HAM10000 lesion classification | 3 active ensemble | 61–62% test | ~59% test | Weak/conservative |
| Chest X-ray | CheXpert multilabel | 1 | sample-level demo only | — | Single pretrained model |

### Binary sensitivity / specificity

| Model | Positive class | Sensitivity / Recall | Specificity |
|---|---|---:|---:|
| Bone X-ray | Abnormal | ~77.4% | ~85.9% |
| Breast Mammography | Malignant | ~73.5% | ~68.4% |

### Policy action distribution

From the combined controlled evaluation:

| Policy action | Count | Meaning |
|---|---:|---|
| ANSWER | 6 | Output allowed |
| ESCALATE | 6 | Model ran, but human review recommended |
| STOP | 41 | Input stopped before unsafe inference |

STOP cases mainly correspond to synthetic/non-medical OOD samples. ESCALATE appears for valid but uncertain medical cases such as skin, retina, and breast.

---

## Main contribution

The main contribution is not only training medical classifiers. The project builds a governed medical AI pipeline that controls whether a prediction should be shown, escalated, refused, or stopped.

Baseline classifier pipeline:

    image → model → prediction

Governed pipeline:

    image
    → medical image gate
    → route detection
    → pre-inference safety gate
    → preprocessing
    → quality check
    → conformal routing
    → model registry
    → specialist inference
    → OOD detection
    → uncertainty estimation
    → explainability
    → governed decision policy
    → ANSWER / ESCALATE / REFUSE / REQUEST_EVIDENCE / STOP

---

## Evaluation status

Current implemented checks:

- Multiclass route detector supports abdomen_ct, brain_mri, bone_xray, breast_mammography, chest_xray, retina_fundus, skin_dermoscopy, and unknown
- Unknown/random/non-medical images are stopped before inference
- All active medical routes run through route detection, preprocessing, model registry resolution, inference, OOD check, explainability, and governed policy
- DICOM test input is converted and passed through the pipeline
- Grad-CAM++ is supported for DenseNet-based and ResNet-based models
- Temporary uploaded files are deleted after processing
- The frontend displays route probabilities, conformal routing candidates, model output probabilities, selected model, inference result, OOD status, policy action, warnings, and heatmap when available
- The frontend supports single review, batch review, manual route confirmation, JSON export, and PDF report export
- The backend exposes route/configuration metadata through `/config` and `/routes`
- Evaluation scripts generate active route, OOD, and final metrics summaries

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
- Output withholding when policy does not allow automatic answer
- OOD method and reason display
- Grad-CAM++ heatmap display when available and policy allows safe display
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

Train one model:

    cd backend
    python3 train_abdomen_ct.py

Train ensemble members if supported by the script:

    python3 train_abdomen_ct.py --seed 42 --output reference_data/models/abdomen/abdomen_seed_42.pth
    python3 train_abdomen_ct.py --seed 123 --output reference_data/models/abdomen/abdomen_seed_123.pth
    python3 train_abdomen_ct.py --seed 777 --output reference_data/models/abdomen/abdomen_seed_777.pth

### Brain MRI model

Dataset:

    Kaggle Brain Tumor MRI Dataset

Expected local path:

    backend/raw_datasets/brain-tumor-mri-dataset/

Train:

    cd backend
    python3 train_brain_mri.py

Ensemble example:

    python3 train_brain_mri.py --seed 42 --output reference_data/models/brain/brain_seed_42.pth
    python3 train_brain_mri.py --seed 123 --output reference_data/models/brain/brain_seed_123.pth
    python3 train_brain_mri.py --seed 777 --output reference_data/models/brain/brain_seed_777.pth

### Bone X-ray model

Dataset:

    MURA-v1.1

Expected local path:

    backend/raw_datasets/MURA-v1.1/

Train ensemble:

    cd backend
    python3 train_bone_xray.py --seed 42 --output reference_data/models/bone/bone_seed_42.pth
    python3 train_bone_xray.py --seed 123 --output reference_data/models/bone/bone_seed_123.pth
    python3 train_bone_xray.py --seed 777 --output reference_data/models/bone/bone_seed_777.pth

### Breast mammography model

Dataset:

    CBIS-DDSM

Expected local path:

    backend/raw_datasets/cbis_ddsm/

Train ensemble:

    cd backend
    python3 train_breast_mammography.py --seed 42 --output reference_data/models/breast/breast_seed_42.pth
    python3 train_breast_mammography.py --seed 123 --output reference_data/models/breast/breast_seed_123.pth
    python3 train_breast_mammography.py --seed 777 --output reference_data/models/breast/breast_seed_777.pth

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

Train ensemble:

    cd backend
    python3 train_retina_fundus.py --seed 42 --output reference_data/models/retina/retina_seed_42.pth
    python3 train_retina_fundus.py --seed 123 --output reference_data/models/retina/retina_seed_123.pth
    python3 train_retina_fundus.py --seed 777 --output reference_data/models/retina/retina_seed_777.pth

---

## Train diffusion OOD detector

The project includes a trained 5-step denoising diffusion OOD detector.

Train:

    cd backend
    python3 train_diffusion_ood.py --epochs 3 --max-images 4000

Saved outputs:

    reference_data/ood/diffusion_ood_model.pth
    reference_data/ood/diffusion_ood_thresholds.json

Calibrate route-specific OOD thresholds:

    cd backend
    python3 calibrate_diffusion_ood_thresholds.py

Saved output:

    reference_data/ood/route_diffusion_thresholds.json

---

## Run evaluation

### Active route evaluation

    cd backend
    source venv/bin/activate
    PYTHONPATH=. python3 evaluation/evaluate_active_routes.py

Saves:

    evaluation/results/active_route_evaluation_<timestamp>.json
    evaluation/results/active_route_evaluation_<timestamp>.csv

### OOD evaluation

    cd backend
    source venv/bin/activate
    PYTHONPATH=. python3 evaluation/evaluate_ood_detector.py

Saves:

    evaluation/results/ood_evaluation_<timestamp>.json
    evaluation/results/ood_evaluation_<timestamp>.csv

### Final metrics summary

    cd backend
    source venv/bin/activate
    PYTHONPATH=. python3 evaluation/build_metrics_summary.py

Saves:

    evaluation/results/final_metrics_summary.txt

---

## Project structure

    medAI_wgyl-ro/
    ├── backend/
    │   ├── main.py
    │   ├── train_route_detector.py
    │   ├── train_abdomen_ct.py
    │   ├── train_brain_mri.py
    │   ├── train_bone_xray.py
    │   ├── train_breast_mammography.py
    │   ├── train_retina_fundus.py
    │   ├── train_diffusion_ood.py
    │   ├── calibrate_diffusion_ood_thresholds.py
    │   ├── evaluation/
    │   │   ├── evaluate_active_routes.py
    │   │   ├── evaluate_ood_detector.py
    │   │   ├── build_metrics_summary.py
    │   │   └── verify_cleanup.py
    │   ├── medalix/
    │   │   ├── api/                   # Orchestrator, routes, pipeline state, session store
    │   │   ├── ingestion/             # Validation, DICOM conversion, medical image gate
    │   │   ├── preprocessing/         # Modality-aware preprocessing
    │   │   ├── detection/             # Route detector and conformal router
    │   │   ├── quality/               # Image/tensor quality assessment
    │   │   ├── ood/                   # OOD detector and pre-inference safety gate
    │   │   ├── registry/              # Model registry and route-to-model resolution
    │   │   ├── inference/             # Specialist inference and deep ensemble logic
    │   │   ├── explainability/        # Grad-CAM++ heatmap generation
    │   │   ├── policy/                # Governed decision policy
    │   │   ├── audit/                 # PHI-free logging and audit traces
    │   │   └── retention/             # Temporary upload deletion
    │   ├── reference_data/
    │   │   ├── model_registry.json
    │   │   ├── policy_thresholds.json
    │   │   ├── route_detector/
    │   │   ├── input_gate/
    │   │   ├── ood/
    │   │   └── models/
    │   └── requirements.txt
    ├── frontend/
    │   └── app/
    │       ├── page.tsx
    │       ├── demo/
    │       ├── review/
    │       └── layout.tsx
    └── README.md

---

## Pipeline stages

Every image passes through the following stages in order:

1. Ingestion validation — file format, header integrity, size limits, and readability checks
2. Format preparation — DICOM files are converted into PNG-compatible image bytes
3. Medical image gate — rejects likely non-medical or unsupported images
4. Route detection — multiclass route detector predicts supported route or unknown
5. Pre-inference safety gate — checks route support, medical confidence, and manual confirmation state
6. Preprocessing — modality-aware resizing, grayscale/RGB handling, and normalization
7. Quality assessment — tensor validity, resolution, contrast warning signals, and corruption indicators
8. Conformal routing — route candidate set and confirmation behavior
9. Model registry — deterministic route-to-model resolution
10. Inference — specialist model inference with deep ensemble or fallback uncertainty estimation
11. OOD detection — trained 5-step denoising diffusion OOD detector with route-specific thresholds
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
- OOD checking is conservative and route-specific, but not a complete medical safety guarantee.
- Manual route confirmation is intended for controlled research/demo review, not casual patient use.
- Some specialist classifiers are intentionally treated conservatively because their reliability is weaker.
- Breast mammography, active skin ensemble, and some retinal severity classes require stronger clinical validation before any real-world use.
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
