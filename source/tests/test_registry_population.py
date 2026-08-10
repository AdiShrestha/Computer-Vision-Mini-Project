"""Verify the populated Lake Registry meets all requirements."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

REGISTRY_DIR = os.path.join(source_root, 'data', 'registry')
REGISTRY_PATH = os.path.join(REGISTRY_DIR, 'lake_registry.json')


def _load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def test_registry_validates_against_schema():
    """Registry passes schema validation."""
    from data.registry.validate_registry import validate
    schema_path = os.path.join(REGISTRY_DIR, 'schema.json')
    valid, errors = validate(REGISTRY_PATH, schema_path)
    assert valid, f"Validation errors: {errors}"


def test_minimum_lake_count():
    """At least 15 lakes total."""
    reg = _load_registry()
    assert len(reg['lakes']) >= 15, f"Only {len(reg['lakes'])} lakes, need >= 15"


def test_south_lhonak_is_evaluation_event():
    """South Lhonak Lake is present and assigned evaluation_event."""
    reg = _load_registry()
    south_lhonak = [l for l in reg['lakes'] 
                    if 'lhonak' in l['name'].lower() or l['id'] == 'SGL-001']
    assert len(south_lhonak) >= 1, "South Lhonak Lake not found"
    assert south_lhonak[0]['role'] == 'evaluation_event'


def test_role_distribution():
    """At least 1 evaluation_event, 3 evaluation_control, 10 training."""
    reg = _load_registry()
    roles = {}
    for lake in reg['lakes']:
        roles[lake['role']] = roles.get(lake['role'], 0) + 1
    
    assert roles.get('evaluation_event', 0) >= 1, "Need >= 1 evaluation_event"
    assert roles.get('evaluation_control', 0) >= 3, "Need >= 3 evaluation_control"
    assert roles.get('training', 0) >= 10, "Need >= 10 training"


def test_unique_ids():
    """All lake IDs are unique."""
    reg = _load_registry()
    ids = [l['id'] for l in reg['lakes']]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {set(x for x in ids if ids.count(x) > 1)}"


def test_no_evaluation_lake_in_training_role():
    """INV-002: No evaluation lake incorrectly assigned training role."""
    reg = _load_registry()
    for lake in reg['lakes']:
        if lake.get('documented_events'):
            assert lake['role'] != 'training', (
                f"Lake {lake['id']} ({lake['name']}) has documented events "
                f"but is assigned 'training' role — potential data leakage"
            )


def test_coordinates_in_hkh_range():
    """All coordinates are within the Hindu Kush Himalaya region."""
    reg = _load_registry()
    for lake in reg['lakes']:
        lat = lake['coordinates']['latitude']
        lon = lake['coordinates']['longitude']
        elev = lake['coordinates']['elevation_m']
        assert 25 <= lat <= 40, f"{lake['id']}: lat {lat} outside HKH range"
        assert 70 <= lon <= 100, f"{lake['id']}: lon {lon} outside HKH range"
        assert 3000 <= elev <= 7000, f"{lake['id']}: elevation {elev}m unusual for glacial lake"


def test_bounding_boxes_consistent():
    """All bounding boxes have north > south and east > west."""
    reg = _load_registry()
    for lake in reg['lakes']:
        bb = lake['bounding_box']
        assert bb['north'] > bb['south'], f"{lake['id']}: north <= south"
        assert bb['east'] > bb['west'], f"{lake['id']}: east <= west"
