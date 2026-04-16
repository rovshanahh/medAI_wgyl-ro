from pathlib import Path
import json
import numpy as np

from medalix.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from medalix.inference.ensemble_model import EnsembleModel


def main():
    image_dir = Path("reference_data/chest_xray_id")
    output_file = Path("reference_data/conformal_scores.json")

    if not image_dir.exists():
        raise FileNotFoundError(f"Folder not found: {image_dir}")

    preprocessor = PreprocessingPipeline()
    model = EnsembleModel()

    scores = []

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        try:
            raw_bytes = image_path.read_bytes()
            _, tensor = preprocessor.run(raw_bytes)
            result = model.predict(tensor)

            probs = result.get("probabilities", {})
            if not probs:
                continue

            top_prob = max(probs.values())
            nonconformity = 1.0 - float(top_prob)
            scores.append(nonconformity)

        except Exception as e:
            print(f"Skipping {image_path.name}: {e}")

    if not scores:
        raise ValueError("No conformal scores computed.")

    scores = sorted(scores)
    alpha = 0.1
    n = len(scores)
    q_index = int(np.ceil((n + 1) * (1 - alpha))) - 1
    q_index = min(max(q_index, 0), n - 1)
    threshold = scores[q_index]

    result = {
        "alpha": alpha,
        "num_samples": n,
        "threshold": threshold,
        "scores": scores,
    }

    output_file.write_text(json.dumps(result, indent=2))
    print(f"Saved conformal scores to {output_file}")
    print(f"Threshold = {threshold:.6f}")


if __name__ == "__main__":
    main()