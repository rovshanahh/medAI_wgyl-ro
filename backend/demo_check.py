import subprocess
import sys
from pathlib import Path


COMMANDS = [
    ("Backend asset check", ["python3", "check_backend_assets.py"]),
    ("Route detector test", ["python3", "test_route_detector.py"]),
    ("Pipeline smoke test", ["python3", "smoke_test_pipeline.py"]),
    ("Active route evaluation", ["python3", "evaluate_active_routes.py"]),
]


def run_command(title: str, command: list[str]) -> bool:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(" ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nFAILED: {title}")
        return False

    print(f"\nPASSED: {title}")
    return True


def main() -> None:
    if not Path("medalix").exists():
        print("Run this script from the backend directory.")
        sys.exit(1)

    passed = 0

    for title, command in COMMANDS:
        if run_command(title, command):
            passed += 1
        else:
            break

    print("\n" + "=" * 90)
    print("DEMO CHECK SUMMARY")
    print("=" * 90)
    print(f"Passed: {passed}/{len(COMMANDS)}")

    if passed != len(COMMANDS):
        sys.exit(1)

    print("All demo checks passed.")


if __name__ == "__main__":
    main()