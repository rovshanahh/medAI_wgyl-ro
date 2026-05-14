from pathlib import Path
from medalix.api.orchestrator import Orchestrator


TEST_FILE = Path("test_samples/chest_xray.jpg")


def main():
    orch = Orchestrator()

    before = set(Path("uploads").rglob("*")) if Path("uploads").exists() else set()

    result = orch.execute(
        filename=TEST_FILE.name,
        content_type="image/jpeg",
        content=TEST_FILE.read_bytes(),
        route_override="chest_xray",
    )

    after = set(Path("uploads").rglob("*")) if Path("uploads").exists() else set()
    new_files = [path for path in after - before if path.is_file()]

    print("analysis_id:", result.get("analysis_id"))
    print("message:", result.get("message"))
    print("policy:", result.get("policy"))
    print("new_upload_files_remaining:", len(new_files))

    if new_files:
        print("Remaining files:")
        for path in new_files:
            print(" -", path)
    else:
        print("Cleanup verified: no new uploaded temp files remain after analysis.")


if __name__ == "__main__":
    main()
