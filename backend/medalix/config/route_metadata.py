ACTIVE_ROUTE_METADATA = [
    {
        "route": "brain_mri",
        "region": "brain",
        "modality": "mri",
        "model": "brain_mri_resnet18",
        "preprocessing_modality": "mri",
        "description": "Reviews brain MRI images and returns the most likely tumor-related class.",
        "status": "ACTIVE",
    },
    {
        "route": "bone_xray",
        "region": "bone",
        "modality": "xray",
        "model": "bone_xray_standard",
        "preprocessing_modality": "xray",
        "description": "Reviews bone X-ray images and separates normal from abnormal cases.",
        "status": "ACTIVE",
    },
    {
        "route": "breast_mammography",
        "region": "breast",
        "modality": "mammography",
        "model": "breast_mammography_resnet18",
        "preprocessing_modality": "mammography",
        "description": "Reviews mammography images and separates benign from malignant cases.",
        "status": "ACTIVE",
    },
    {
        "route": "chest_xray",
        "region": "chest",
        "modality": "xray",
        "model": "chest_xray_mvp",
        "preprocessing_modality": "xray",
        "description": "Reviews chest X-ray images and highlights possible visible findings.",
        "status": "ACTIVE",
    },
    {
        "route": "retina_fundus",
        "region": "retina",
        "modality": "fundus",
        "model": "retina_fundus_resnet18",
        "preprocessing_modality": "fundus",
        "description": "Reviews eye fundus images and estimates diabetic retinopathy severity.",
        "status": "ACTIVE",
    },
    {
        "route": "skin_dermoscopy",
        "region": "skin",
        "modality": "dermoscopy",
        "model": "skin_dermoscopy_resnet18",
        "preprocessing_modality": "dermoscopy",
        "description": "Reviews skin dermoscopy images and returns the most likely lesion class.",
        "status": "ACTIVE",
    },
]

SAFETY_ROUTES = [
    {
        "route": "unknown",
        "region": None,
        "modality": None,
        "model": None,
        "description": "Stops unsupported or uncertain images before inference.",
        "status": "STOP_BEFORE_INFERENCE",
    }
]

INACTIVE_PLACEHOLDERS = [
    {
        "route": "abdomen_ct",
        "region": "abdomen",
        "modality": "ct",
        "status": "INACTIVE",
    },
]

ROUTE_TO_REGION_MODALITY = {
    item["route"]: (item["region"], item["modality"])
    for item in ACTIVE_ROUTE_METADATA
}

ROUTE_TO_REGION_MODALITY["unknown"] = (None, None)

ROUTE_TO_PREPROCESSING_MODALITY = {
    item["route"]: item["preprocessing_modality"]
    for item in ACTIVE_ROUTE_METADATA
}