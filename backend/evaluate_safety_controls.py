from pathlib import Path

from medalix.api.orchestrator import Orchestrator


TEST_CASES = [
    {
        "name": "valid_brain_mri",
        "path": "test_samples/brain_mri.jpg",
        "expected_route": "brain_mri",
        "expected_policy": ["ANSWER", "ESCALATE"],
        "override": None,
    },
    {
        "name": "valid_chest_xray",
        "path": "test_samples/chest_xray.jpg",
        "expected_route": "chest_xray",
        "expected_policy": ["ANSWER", "ESCALATE"],
        "override": None,
    },
    {
        "name": "unknown_random_image",
        "path": "test_samples/random.jpg",
        "expected_route": "unknown",
        "expected_policy": ["STOP"],
        "override": None,
    },
    {
        "name": "manual_override_random_to_chest",
        "path": "test_samples/random.jpg",
        "expected_route": "chest_xray",
        "expected_policy": ["ANSWER", "ESCALATE", "STOP"],
        "override": "chest_xray",
    },
]


def percent(value):
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def safe_get(data, *keys, default=None):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def evaluate_case(orchestrator: Orchestrator, case: dict) -> dict:
    path = Path(case["path"])

    if not path.exists():
        return {
            "name": case["name"],
            "status": "MISSING",
            "reason": f"File not found: {path}",
        }

    result = orchestrator.execute(
        filename=path.name,
        content_type=None,
        content=path.read_bytes(),
        route_override=case.get("override"),
    )

    selected_route = safe_get(result, "input_gate", "selected_route", default="unknown")
    policy_action = safe_get(result, "policy", "action", default="—")
    top_label = safe_get(result, "inference", "top_label", default="No inference")
    confidence = safe_get(result, "inference", "top_probability", default=None)
    manual_override = safe_get(result, "input_gate", "manual_override", default=False)
    heatmap_path = safe_get(result, "explainability", "heatmap_path", default=None)

    route_ok = selected_route == case["expected_route"]
    policy_ok = policy_action in case["expected_policy"]

    return {
        "name": case["name"],
        "status": "PASS" if route_ok and policy_ok else "FAIL",
        "selected_route": selected_route,
        "expected_route": case["expected_route"],
        "policy_action": policy_action,
        "expected_policy": case["expected_policy"],
        "output": top_label,
        "confidence": confidence,
        "manual_override": manual_override,
        "heatmap": bool(heatmap_path),
        "message": result.get("message"),
    }


def main():
    orchestrator = Orchestrator()
    results = [evaluate_case(orchestrator, case) for case in TEST_CASES]

    print("\n" + "=" * 110)
    print("SAFETY CONTROL EVALUATION")
    print("=" * 110)

    print(
        f"{'Case':30} | {'Status':6} | {'Route':18} | {'Policy':10} | "
        f"{'Output':22} | {'Conf.':8} | {'Override':8} | {'Heatmap'}"
    )
    print("-" * 110)

    for item in results:
        print(
            f"{item['name'][:30]:30} | "
            f"{item['status'][:6]:6} | "
            f"{item.get('selected_route', '—')[:18]:18} | "
            f"{item.get('policy_action', '—')[:10]:10} | "
            f"{item.get('output', '—')[:22]:22} | "
            f"{percent(item.get('confidence')):8} | "
            f"{str(item.get('manual_override', False)):8} | "
            f"{'Yes' if item.get('heatmap') else 'No'}"
        )

    print("=" * 110)

    passed = sum(1 for item in results if item["status"] == "PASS")
    failed = sum(1 for item in results if item["status"] == "FAIL")
    missing = sum(1 for item in results if item["status"] == "MISSING")

    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Missing: {missing}")

    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()