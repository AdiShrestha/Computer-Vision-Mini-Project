"""Verify Lake Registry schema and validation tooling."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

REGISTRY_DIR = os.path.join(source_root, 'data', 'registry')


def test_schema_is_valid_json():
    """schema.json is valid JSON."""
    schema_path = os.path.join(REGISTRY_DIR, 'schema.json')
    with open(schema_path) as f:
        schema = json.load(f)
    assert '$schema' in schema
    assert 'lakes' in schema['properties']


def test_validate_correct_registry():
    """Validation passes on the test registry."""
    from data.registry.validate_registry import validate
    registry_path = os.path.join(REGISTRY_DIR, 'test_registry.json')
    schema_path = os.path.join(REGISTRY_DIR, 'schema.json')
    valid, errors = validate(registry_path, schema_path)
    assert valid, f"Validation failed: {errors}"


def test_validate_catches_duplicate_ids():
    """Validation catches duplicate lake IDs."""
    import tempfile
    from data.registry.validate_registry import validate
    schema_path = os.path.join(REGISTRY_DIR, 'schema.json')
    
    # Create a registry with duplicate IDs
    with open(os.path.join(REGISTRY_DIR, 'test_registry.json')) as f:
        reg = json.load(f)
    if len(reg['lakes']) >= 2:
        reg['lakes'][1]['id'] = reg['lakes'][0]['id']
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(reg, f)
        tmp_path = f.name
    
    try:
        valid, errors = validate(tmp_path, schema_path)
        assert not valid, "Should have caught duplicate IDs"
        assert any('duplicate' in e.lower() for e in errors)
    finally:
        os.unlink(tmp_path)


def test_validate_catches_invalid_bbox():
    """Validation catches north < south in bounding box."""
    import tempfile
    from data.registry.validate_registry import validate
    schema_path = os.path.join(REGISTRY_DIR, 'schema.json')
    
    with open(os.path.join(REGISTRY_DIR, 'test_registry.json')) as f:
        reg = json.load(f)
    # Swap north and south
    lake = reg['lakes'][0]
    lake['bounding_box']['north'], lake['bounding_box']['south'] = (
        lake['bounding_box']['south'], lake['bounding_box']['north']
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(reg, f)
        tmp_path = f.name
    
    try:
        valid, errors = validate(tmp_path, schema_path)
        assert not valid, "Should have caught invalid bounding box"
    finally:
        os.unlink(tmp_path)


def test_validate_requires_evaluation_event():
    """Validation requires at least one evaluation_event lake."""
    import tempfile
    from data.registry.validate_registry import validate
    schema_path = os.path.join(REGISTRY_DIR, 'schema.json')
    
    with open(os.path.join(REGISTRY_DIR, 'test_registry.json')) as f:
        reg = json.load(f)
    # Remove all evaluation_event lakes
    for lake in reg['lakes']:
        if lake['role'] == 'evaluation_event':
            lake['role'] = 'training'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(reg, f)
        tmp_path = f.name
    
    try:
        valid, errors = validate(tmp_path, schema_path)
        assert not valid, "Should require at least one evaluation_event lake"
    finally:
        os.unlink(tmp_path)
