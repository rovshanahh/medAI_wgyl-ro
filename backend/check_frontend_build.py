import subprocess
import sys
from pathlib import Path


FRONTEND_DIR = Path("../frontend")


def main() -> None:
    if not FRONTEND_DIR.exists():
        print("Frontend directory not found.")
        sys.exit(1)

    print("\n" + "=" * 90)
    print("FRONTEND BUILD CHECK")
    print("=" * 90)

    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=FRONTEND_DIR,
    )

    if result.returncode != 0:
        print("\nFrontend build failed.")
        sys.exit(result.returncode)

    print("\nFrontend build passed.")


if __name__ == "__main__":
    main()