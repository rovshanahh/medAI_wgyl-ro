import json
from pathlib import Path

import torch

from medalix.api.orchestrator import Orchestrator


REFERENCE_IMAGES = {
    "abdomen_ct": [
        "test_samples/abdomen_ct.jpg",
    ],
    "brain_mri": [
        "test_samples/brain_mri.jpg",
    ],
    "bone_xray": [
        "test_samples/bone_xray.png",
    ],
    "breast_mammography": [
        "test_samples/breast_mammography.jpg",
    ],
    "chest_xray": [
        "test_samples/chest_xray.jpg",
        "test_samples/test_chest_xray.dcm",
    ],
    "retina_fundus": [
        "test_samples/retina_fundus.jpg",
    ],
    "skin_dermoscopy": [
        "test_samples/skin_dermoscopy.jpg",
    ],
}

OUTPUT_PATH = Path("reference_data/ood/feature_stats.json")


def collect_features(orchestrator: Orchestrator, route: str, paths: list[str]) -> list[list[float]]:
    features = []

    for item in paths:
        path = Path(item)

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        result = orchestrator.execute(
            filename=path.name,
            content_type=None,
            content=path.read_bytes(),
        )

        selected_route = result.get("input_gate", {}).get("selected_route")
        inference = result.get("inference", {})
        feature_vector = inference.get("features")

        if selected_route != route:
            print(f"Skipping {path}; expected {route}, got {selected_route}")
            continue

        if not feature_vector:
            print(f"Skipping {path}; no feature vector available")
            continue

        features.append(feature_vector)

    return features


def build_stats(features: list[list[float]]) -> dict | None:
    if not features:
        return None

    tensor = torch.tensor(features, dtype=torch.float32)

    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0, unbiased=False)

    std = torch.clamp(std, min=1e-6)

    return {
        "count": len(features),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold_near": 2.5,
        "threshold_hard": 4.0,
    }


def main() -> None:
    orchestrator = Orchestrator()
    all_stats = {}

    for route, paths in REFERENCE_IMAGES.items():
        print(f"Building OOD feature stats for: {route}")
        features = collect_features(orchestrator, route, paths)
        stats = build_stats(features)

        if stats is None:
            print(f"  No stats generated for {route}")
            continue

        all_stats[route] = stats
        print(f"  Saved stats with {stats['count']} reference feature vector(s)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")

    print()
    print(f"Saved OOD feature stats to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
