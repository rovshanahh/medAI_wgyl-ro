from pathlib import Path
import json
import numpy as np

from medalix.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from medalix.inference.ensemble_model import EnsembleModel


def main():
    image_dir = Path("reference_data/chest_xray_id")
    output_file = Path("reference_data/feature_stats.json")

    if not image_dir.exists():
        raise FileNotFoundError(f"Folder not found: {image_dir}")

    preprocessor = PreprocessingPipeline()
    model = EnsembleModel()

    feature_list = []

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        try:
            raw_bytes = image_path.read_bytes()
            _, tensor = preprocessor.run(raw_bytes)
            result = model.predict(tensor)

            features = result.get("features")

            print("FILE:", image_path.name)
            print("RESULT KEYS:", list(result.keys()))
            print("HAS FEATURES:", features is not None)

            if features is not None:
                feature_list.append(features)

        except Exception as e:
            print(f"Skipping {image_path.name}: {e}")

    if not feature_list:
        raise ValueError("No valid feature vectors found.")

    features_array = np.array(feature_list, dtype=np.float32)

    stats = {
        "num_samples": int(features_array.shape[0]),
        "feature_dim": int(features_array.shape[1]),
        "mean_vector": features_array.mean(axis=0).tolist(),
        "std_vector": features_array.std(axis=0).tolist(),
    }

    output_file.write_text(json.dumps(stats, indent=2))
    print(f"Saved feature stats to {output_file}")


if __name__ == "__main__":
    main()