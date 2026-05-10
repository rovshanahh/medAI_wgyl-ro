ACTIVE_ROUTE_METADATA = [
    {
        "route": "abdomen_ct",
        "region": "abdomen",
        "modality": "ct",
        "model": "abdomen_ct_resnet18",
        "description": "Reviews kidney CT images and classifies cyst, normal, stone, or tumor cases.",
        "status": "ACTIVE",
        "preprocessing_modality": "ct",
    },
    {
        "route": "bone_xray",
        "region": "bone",
        "modality": "xray",
        "model": "bone_xray_standard",
        "description": "Reviews bone X-ray images and separates normal from abnormal cases.",
        "status": "ACTIVE",
        "preprocessing_modality": "xray",
    },
    {
        "route": "brain_mri",
        "region": "brain",
        "modality": "mri",
        "model": "brain_mri_resnet18",
        "description": "Reviews brain MRI images and returns the most likely tumor-related class.",
        "status": "ACTIVE",
        "preprocessing_modality": "mri",
    },
    {
        "route": "breast_mammography",
        "region": "breast",
        "modality": "mammography",
        "model": "breast_mammography_resnet18",
        "description": "Reviews mammography images and separates benign from malignant cases.",
        "status": "ACTIVE",
        "preprocessing_modality": "mammography",
    },
    {
        "route": "chest_xray",
        "region": "chest",
        "modality": "xray",
        "model": "chest_xray_mvp",
        "description": "Reviews chest X-ray images and highlights possible visible findings.",
        "status": "ACTIVE",
        "preprocessing_modality": "xray",
    },
    {
        "route": "retina_fundus",
        "region": "retina",
        "modality": "fundus",
        "model": "retina_fundus_resnet18",
        "description": "Reviews eye fundus images and estimates diabetic retinopathy severity.",
        "status": "ACTIVE",
        "preprocessing_modality": "fundus",
    },
    {
        "route": "skin_dermoscopy",
        "region": "skin",
        "modality": "dermoscopy",
        "model": "skin_dermoscopy_resnet18",
        "description": "Reviews skin dermoscopy images and returns the most likely lesion class.",
        "status": "ACTIVE",
        "preprocessing_modality": "dermoscopy",
    },
]


SAFETY_ROUTES = [
    {
        "route": "unknown",
        "description": "Stops unsupported or uncertain inputs before inference.",
        "status": "SAFETY",
    }
]


INACTIVE_PLACEHOLDERS = []


ROUTE_TO_REGION_MODALITY = {
    item["route"]: (item["region"], item["modality"])
    for item in ACTIVE_ROUTE_METADATA
}


ROUTE_TO_PREPROCESSING_MODALITY = {
    item["route"]: item["preprocessing_modality"]
    for item in ACTIVE_ROUTE_METADATA
}


SUPPORTED_ROUTE_LABELS = [item["route"] for item in ACTIVE_ROUTE_METADATA]


ALL_ROUTE_LABELS = SUPPORTED_ROUTE_LABELS + ["unknown"]