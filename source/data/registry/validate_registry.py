"""Lake Registry Validation Tool.

Validates any registry JSON against schema.json and performs additional custom checks:
- Unique lake IDs
- At least one lake with role 'evaluation_event'
- Bounding box consistency (north > south, east > west)
- Geographic independence check (INV-002: warning if training & evaluation share basin)
"""
import os
import sys
import json
import argparse
from typing import Tuple, List, Dict, Any
import jsonschema


def validate(registry_path: str, schema_path: str) -> Tuple[bool, List[str]]:
    """Validate a registry JSON file against schema.json and custom rules.

    Args:
        registry_path: Path to lake registry JSON file.
        schema_path: Path to schema.json file.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_error_and_warning_messages)
    """
    errors: List[str] = []

    if not os.path.exists(registry_path):
        return False, [f"Registry file not found: {registry_path}"]

    if not os.path.exists(schema_path):
        return False, [f"Schema file not found: {schema_path}"]

    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Registry file is not valid JSON: {e}"]

    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Schema file is not valid JSON: {e}"]

    # 1. JSON Schema validation
    validator = jsonschema.Draft7Validator(schema)
    for err in validator.iter_errors(registry):
        errors.append(f"Schema Error at {'.'.join(str(p) for p in err.path)}: {err.message}")

    if errors:
        return False, errors

    # 2. Custom Semantic Validation
    lakes = registry.get("lakes", [])

    # Check 2a: Unique Lake IDs
    seen_ids = set()
    for lake in lakes:
        lake_id = lake.get("id")
        if lake_id in seen_ids:
            errors.append(f"Duplicate lake ID found: {lake_id}")
        seen_ids.add(lake_id)

    # Check 2b: At least one evaluation_event lake
    eval_event_lakes = [l for l in lakes if l.get("role") == "evaluation_event"]
    if not eval_event_lakes:
        errors.append("Registry must contain at least one lake with role 'evaluation_event'.")

    # Check 2c: Bounding box consistency (north > south, east > west)
    for lake in lakes:
        bbox = lake.get("bounding_box", {})
        north = bbox.get("north")
        south = bbox.get("south")
        east = bbox.get("east")
        west = bbox.get("west")

        if north is not None and south is not None and north <= south:
            errors.append(f"Lake {lake.get('id')}: invalid bounding box north ({north}) <= south ({south}).")
        if east is not None and west is not None and east <= west:
            errors.append(f"Lake {lake.get('id')}: invalid bounding box east ({east}) <= west ({west}).")

    # Check 2d: INV-002 Geographic Independence Warning
    training_basins = {l.get("basin") for l in lakes if l.get("role") == "training" and l.get("basin")}
    eval_basins = {l.get("basin") for l in lakes if l.get("role") in ("evaluation_event", "evaluation_control") and l.get("basin")}
    shared_basins = training_basins.intersection(eval_basins)
    if shared_basins:
        print(f"WARNING (INV-002): Training and evaluation lakes share basin(s): {shared_basins}")

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    parser = argparse.ArgumentParser(description="Validate Lake Registry JSON against schema.json.")
    parser.add_argument("registry_path", help="Path to lake_registry.json")
    parser.add_argument("--schema", default=None, help="Path to schema.json (default: source/data/registry/schema.json)")
    args = parser.parse_args()

    if args.schema is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(script_dir, "schema.json")
    else:
        schema_path = args.schema

    valid, errors = validate(args.registry_path, schema_path)
    if valid:
        print(f"PASS: {args.registry_path} is valid.")
        sys.exit(0)
    else:
        print(f"FAIL: {args.registry_path} is invalid:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
