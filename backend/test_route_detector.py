from pathlib import Path

from medalix.detection.route_detector import RouteDetector


TEST_IMAGES = [
    "test_samples/brain_mri.jpg",
    "test_samples/bone_xray.png",
    "test_samples/chest_xray.jpg",
    "test_samples/random.jpg",
    "test_samples/retina_fundus.jpg",
    "test_samples/skin_dermoscopy.jpg",
]


def main() -> None:
    detector = RouteDetector()

    for image_path in TEST_IMAGES:
        path = Path(image_path)

        if not path.exists():
            print(f"\nMissing: {image_path}")
            continue

        result = detector.predict(path.read_bytes())

        print("\nImage:", image_path)
        print("Route:", result["route_label"])
        print("Raw route:", result["raw_route_label"])
        print("Region:", result["region"])
        print("Modality:", result["modality"])
        print("Confidence:", round(result["confidence"], 4))
        print("Margin:", round(result["margin"], 4))
        print("Supported:", result["supported"])
        print("Requires confirmation:", result["requires_confirmation"])
        print("Reason:", result["reason"])
        print("Probabilities:")

        for label, prob in result["probabilities"].items():
            print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()