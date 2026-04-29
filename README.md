# MedAIx

A research-use medical imaging AI platform built for TED University SENG 492 Senior Project.

MedAIx accepts medical images, validates them through a multi-gate input pipeline, routes them to the correct specialist model, runs deep ensemble inference with uncertainty quantification, generates Grad-CAM++ visual explanations, and returns a governed policy decision through a single end-to-end pipeline.

This project is for research and educational use only. Outputs are non-diagnostic and must not be used for clinical decision-making.

---

## What it does

- Accepts uploaded images through the web interface (JPEG, PNG, TIFF, DICOM)
- Rejects non-medical and out-of-distribution inputs before analysis
- Routes supported images to one of three active pipelines:
  - Chest X-ray — 14-label multilabel classification (DenseNet121, CheXpert)
  - Bone X-ray — Normal / Abnormal binary classification (DenseNet121, MURA)
  - Brain MRI — 4-class tumor classification: Glioma, Meningioma, No Tumor, Pituitary (ResNet18)
- Runs MC dropout ensemble inference with epistemic and aleatoric uncertainty quantification
- Generates Grad-CAM++ heatmaps for all three routes
- Evaluates all pipeline signals through a governed decision policy: ANSWER / REFUSE / ESCALATE / REQUEST_EVIDENCE / STOP
- Returns a structured response including input gate result, detected region and modality, selected route and model, quality check, OOD status, uncertainty metrics, explainability output, and policy decision

---

## Supported routes

| Route | Gate | Inference | Heatmap | Status |
|-------|------|-----------|---------|--------|
| Chest X-ray | Active | 14-label multilabel | Grad-CAM++ | Active |
| Bone X-ray | Active | Normal / Abnormal | Grad-CAM++ | Active |
| Brain MRI | Active | 4-class tumor | Grad-CAM++ | Active |
| Abdomen CT | — | — | — | Placeholder |
| Breast mammography | — | — | — | Placeholder |
| Skin dermoscopy | — | — | — | Placeholder |
| Retina fundus | — | — | — | Placeholder |

---

## Evaluation results

Chest X-ray pipeline (20-image evaluation set):
- Chest images accepted (not STOP): 20/20 — 100%
- Non-chest images rejected (STOP): 20/20 — 100%

Brain MRI model (Kaggle Brain Tumor MRI Dataset, independent test set):
- Validation accuracy: 95.3% across 4 classes (Glioma, Meningioma, No Tumor, Pituitary)

---

## Run the project

### Requirements

- Python 3.11 or later
- Node.js 18 or later
- macOS or Linux

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p logs/heatmaps temp_uploads
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000

Health check: http://localhost:8000/health

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:3000

Make sure the backend is running before using the frontend.

---

## Train models

Model checkpoint files are not stored in this repository due to size. Run the following scripts to generate them locally.

```bash
cd backend

# Chest X-ray input gate
python3 train_input_gate.py

# Bone X-ray input gate
python3 train_bone_input_gate.py

# Medical X-ray top-level gate
python3 train_medical_xray_gate.py

# Brain MRI input gate
python3 train_brain_mri_gate.py

# Brain MRI inference model (requires Kaggle brain tumor dataset)
python3 train_brain_mri.py
```

Brain MRI training dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Place the dataset at: ~/Downloads/brain-tumor-mri-dataset/

---

## Run evaluation

```bash
cd backend
source venv/bin/activate
python3 evaluate_chest_mvp.py
```

Results are saved to evaluation/evaluation_results.json.

---

## Project structure

```
medAI_wgyl-ro/
├── backend/
│   ├── main.py
│   ├── medalix/
│   │   ├── api/                   # Orchestrator, routes, session store
│   │   ├── ingestion/             # Input gates and validation
│   │   ├── preprocessing/         # Modality-aware preprocessing
│   │   ├── ood/                   # OOD detection
│   │   ├── detection/             # Region and modality detection, conformal routing
│   │   ├── quality/               # Image quality assessment
│   │   ├── registry/              # Model registry and routing table
│   │   ├── inference/             # Ensemble model with MC dropout
│   │   ├── explainability/        # Grad-CAM++ heatmap generation
│   │   ├── policy/                # Governed decision policy
│   │   ├── audit/                 # PHI-free logging and audit traces
│   │   └── retention/             # 60-second image deletion policy
│   ├── reference_data/
│   │   ├── model_registry.json
│   │   ├── policy_thresholds.json
│   │   ├── conformal_scores.json
│   │   └── feature_stats.json
│   └── evaluation/
├── frontend/
│   └── app/
│       └── page.tsx
└── README.md
```

---

## Pipeline stages

Every image passes through the following stages in order:

1. Ingestion validation — file format, header integrity, size limits
2. Input gate — multi-gate binary classifiers (chest, bone, brain, medical xray)
3. Preprocessing — modality-aware normalization and resizing
4. Quality assessment — blur, noise, artifact, orientation checks
5. OOD detection — feature distance from reference distribution
6. Region and modality detection — conformal prediction routing (alpha = 0.1)
7. Model registry — deterministic route-to-model resolution
8. Ensemble inference — MC dropout with epistemic and aleatoric uncertainty
9. Explainability — Grad-CAM++ heatmap generation
10. Governed decision policy — multi-gate abstention policy

---

## Team

Gular Haji-Hasanli — Software Engineering, TED University
Rovshana Haji-Hasanli — Software Engineering, TED University
Arya Tabiyehzad — Computer Engineering, TED University

---

## Important

This platform is intended solely for research and educational use.
Outputs are non-diagnostic and must not be used for clinical decision-making.
