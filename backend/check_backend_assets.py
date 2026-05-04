import json
from pathlib import Path


REGISTRY_PATH = Path("reference_data/model_registry.json")


def check_file(path_value: str) -> dict:
    if not path_value:
        return {
            "path": "",
            "exists": True,
            "note": "No local file required",
        }

    path = Path(path_value)

    return {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
    }


def main() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    print("\n" + "=" * 90)
    print("BACKEND ASSET CHECK")
    print("=" * 90)
    print(f"Registry version: {registry.get('registry_version')}")

    models = registry.get("models", {})
    routes = registry.get("routes", [])

    print("\nRoutes:")
    for route in routes:
        print(
            f"- {route.get('region')}:{route.get('modality')} "
            f"-> {route.get('model_id')}"
        )

    print("\nModels:")
    missing = []

    for model_id, model in models.items():
        status = model.get("status")
        architecture = model.get("architecture")
        model_path = model.get("model_path", "")
        file_check = check_file(model_path)

        exists_label = "OK" if file_check["exists"] else "MISSING"

        if not file_check["exists"]:
            missing.append(model_id)

        size_label = ""
        if file_check.get("size_mb") is not None:
            size_label = f" ({file_check['size_mb']} MB)"

        print(
            f"- {model_id} | {status} | {architecture} | "
            f"{exists_label} {file_check['path']}{size_label}"
        )

    print("\n" + "=" * 90)

    if missing:
        print("Missing model files:")
        for model_id in missing:
            print(f"- {model_id}")
        raise SystemExit(1)

    print("All required backend assets are present.")


if __name__ == "__main__":
    main()