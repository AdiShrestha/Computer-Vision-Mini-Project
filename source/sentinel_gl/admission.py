"""Content and ancestry admission for measurements and derived artifacts.

Structural admission does not authenticate a provider or establish scientific
validity. Observational decoding remains blocked pending independently reviewed
geospatial adapters. Analytical fixtures use a separate, explicit origin.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import math
import re
from .integrity import digest, loads_strict, verified_file

SCHEMAS = Path(__file__).with_name('schemas')
QUARANTINED = frozenset(loads_strict((SCHEMAS / 'excluded_digests.json').read_text())['sha256'])


class AdmissionError(ValueError):
    """An input is missing, inconsistent, excluded, or unsupported."""


def blocked(operation):
    raise AdmissionError(f'BLOCKED: {operation} requires reviewed measurement, scope, and task adapters')


def schema_check(value, schema):
    """Validate the deliberately limited schema vocabulary shipped here."""
    if "$ref" in schema:
        if set(schema) != {"$ref"} or not isinstance(schema["$ref"], str):
            raise AdmissionError("invalid schema reference")
        name = schema["$ref"]
        if Path(name).name != name or not name.endswith(".json"):
            raise AdmissionError("unsafe schema reference")
        schema_check(value, loads_strict((SCHEMAS / name).read_text()))
        return
    supported = {'type', 'properties', 'required', 'additionalProperties', 'items',
                 'minItems', 'enum', 'const', 'minLength', 'pattern', 'minimum'}
    if set(schema) - supported:
        raise AdmissionError('unsupported schema keyword')
    kind = schema.get('type')
    kinds = kind if isinstance(kind, list) else [kind]
    types = {'object': dict, 'array': list, 'string': str, 'integer': int,
             'number': (int, float), 'boolean': bool, 'null': type(None)}
    if kind is not None and not any(type(value) in (types[k] if isinstance(types[k], tuple) else (types[k],)) for k in kinds):
        raise AdmissionError('schema type confusion')
    if type(value) in (int, float) and not math.isfinite(value):
        raise AdmissionError('nonfinite number')
    if 'const' in schema and (type(value) is not type(schema['const']) or value != schema['const']):
        raise AdmissionError('schema constant mismatch')
    if 'enum' in schema and value not in schema['enum']:
        raise AdmissionError('schema enum mismatch')
    if isinstance(value, dict):
        if not set(schema.get('required', [])) <= value.keys():
            raise AdmissionError('missing schema field')
        for key, child in value.items():
            rule = schema.get('properties', {}).get(key, schema.get('additionalProperties', True))
            if rule is False:
                raise AdmissionError('unknown schema field: ' + key)
            if isinstance(rule, dict):
                schema_check(child, rule)
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0):
            raise AdmissionError('empty required membership')
        for child in value:
            schema_check(child, schema.get('items', {}))
    if isinstance(value, str):
        if len(value.strip()) < schema.get('minLength', 0) or ('pattern' in schema and re.fullmatch(schema['pattern'], value) is None):
            raise AdmissionError('invalid schema string')
    if type(value) in (int, float) and value < schema.get('minimum', value):
        raise AdmissionError('below minimum')


def validate_schema(value, name):
    # Round-trip also checks nested JSON numbers, dictionary keys and builtin types.
    digest(value)
    schema_check(value, loads_strict((SCHEMAS / (name + '.json')).read_text()))


def timestamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise AdmissionError('timezone required')
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class Absence:
    status: str
    reason: str
    value: None = None


class AdmissionContext:
    """A current byte-root plus an independently supplied source manifest.

    The manifest is a trust input, never a proof of provider authenticity. The
    caller must retain its independent review. No mutable admitted token escapes:
    every read traverses and verifies the complete current ancestry again.
    """
    def __init__(self, root, manifest, *, mode='observational'):
        if mode not in ('observational', 'fixture', 'simulation'):
            raise AdmissionError('unknown origin policy')
        validate_schema(manifest, 'manifest')
        self.root = Path(root)
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise AdmissionError("symlink evidence root")
        self.manifest = loads_strict(__import__('json').dumps(manifest, allow_nan=False))
        self.mode = mode
        for key, entry in self.manifest['artifacts'].items():
            if key != entry['artifact_id']:
                raise AdmissionError('artifact ID mismatch')
        paths = [e['path'] for e in self.manifest['artifacts'].values()]
        if len(paths) != len(set(paths)):
            raise AdmissionError('duplicate artifact path')

    def artifact(self, artifact_id, trail=()):
        if artifact_id in trail or len(trail) > 64:
            raise AdmissionError('cyclic or excessive ancestry')
        entry = self.manifest['artifacts'].get(artifact_id)
        if entry is None:
            raise AdmissionError('unknown artifact ID: ' + artifact_id)
        if entry['origin'] != self.mode:
            raise AdmissionError('fixture/simulation/unknown origin cannot enter observational lineage')
        if len(entry['parents']) != len(set(entry['parents'])):
            raise AdmissionError('duplicate parents')
        if entry['sha256'] in QUARANTINED:
            raise AdmissionError('quarantined content, including renamed bytes')
        parts = Path(entry['path']).parts
        if any(p in {'.migration', 'project', 'factory', '.git'} for p in parts) or parts[:2] == ('docs', 'audit'):
            raise AdmissionError('excluded evidence/recovery path')
        path = verified_file(self.root, entry['path'], entry['sha256'])
        for parent in entry['parents']:
            self.artifact(parent, (*trail, artifact_id))
        return path

    def json(self, artifact_id):
        return loads_strict(self.artifact(artifact_id).read_text())

    def csv(self, artifact_id):
        reader = csv.DictReader(io.StringIO(self.artifact(artifact_id).read_text()))
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AdmissionError('missing or duplicate CSV headers')
        rows = list(reader)
        if any(None in r or any(v is None for v in r.values()) for r in rows):
            raise AdmissionError('malformed CSV row')
        return rows

    def measurement(self, record):
        validate_schema(record, 'measurement')
        if record['origin'] != self.mode:
            raise AdmissionError('measurement origin mismatch')
        if record['schema_version'] == 1 and 'native_sar' in record:
            raise AdmissionError('legacy v1 measurement cannot carry native SAR metadata')
        if record['schema_version'] == 2:
            if record.get('product') != 'SENTINEL-1_GRD_NATIVE' or 'native_sar' not in record:
                raise AdmissionError('v2 measurement requires the native SAR product contract')
            from .ingestion.sentinel1_calibration import CalibrationError, validate_native_sar_metadata
            try:
                native = validate_native_sar_metadata(record['native_sar'],
                    measurement_id=record['measurement_id'], product_id=record['product_id'],
                    artifact_manifest=self.manifest, expected_origin=record['origin'])
            except CalibrationError as exc:
                raise AdmissionError(str(exc)) from exc
        else:
            native = None
        if self.manifest['measurements'].get(record['measurement_id']) != digest(record):
            raise AdmissionError('missing or mismatched measurement membership')
        if record['status'] != 'MEASURED':
            return Absence('NOT_MEASURED', record['status'])
        required_ids = [record[k] for k in ('metadata_id', 'product_id', 'query_id', 'response_id', 'roi_id', 'qa_id')]
        if len(set(required_ids)) != len(required_ids):
            raise AdmissionError('metadata/product/request/ROI/QA IDs must be separate')
        for key in required_ids:
            self.artifact(key)
        if native is not None:
            for role in native['roles'].values():
                self.artifact(role['artifact_id'])
        start, end, available = [timestamp(record[k]) for k in ('support_start', 'support_end', 'available_at')]
        if end < start or available < end:
            raise AdmissionError('support not closed or availability precedes support')
        product = self.manifest['artifacts'][record['product_id']]
        required_parents = set(required_ids) - {record['product_id']}
        if native is not None:
            required_parents.update(role['artifact_id'] for role in native['roles'].values()
                                    if role['artifact_id'] != record['product_id'])
        if not required_parents <= set(product['parents']):
            raise AdmissionError('product missing source contributions')
        metadata = self.json(record['metadata_id'])
        expected = {k: record[k] for k in ('product', 'version', 'endpoint', 'units', 'crs', 'support_start', 'support_end', 'available_at', 'availability_policy')}
        if native is not None:
            expected['native_sar'] = record['native_sar']
        if metadata != expected:
            raise AdmissionError('product metadata linkage mismatch')
        query, response = self.json(record['query_id']), self.json(record['response_id'])
        if query != {'endpoint': record['endpoint'], 'roi_id': record['roi_id'], 'product': record['product']} or response != {'query_id': record['query_id'], 'product_id': record['product_id'], 'status': 'SUCCESS'}:
            raise AdmissionError('query/response linkage mismatch')
        roi = self.json(record['roi_id'])
        if set(roi) != {'type', 'coordinates', 'crs'} or roi['type'] != 'Polygon' or roi['crs'] != record['crs']:
            raise AdmissionError('polygon/CRS missing')
        rings = roi['coordinates']
        if not isinstance(rings, list) or not rings:
            raise AdmissionError('empty polygon')
        for ring in rings:
            if not isinstance(ring, list) or len(ring) < 4 or ring[0] != ring[-1]:
                raise AdmissionError('unclosed polygon')
            for xy in ring:
                if not isinstance(xy, list) or len(xy) != 2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in xy):
                    raise AdmissionError('invalid polygon coordinate')
        qa = self.json(record['qa_id'])
        if qa != {'status': 'ACCEPTED', 'policy': 'analytical-mask-v1'}:
            # Production QA decoding must be independently reviewed before admission.
            raise AdmissionError('unknown/unsupported measurement QA')
        if self.mode != 'fixture':
            raise AdmissionError('observational/simulation measurement decoder pending independent review')
        return record

    def derived(self, record):
        validate_schema(record, 'derived')
        if record['origin'] != self.mode:
            raise AdmissionError('derived origin mismatch')
        self.artifact(record['artifact_id'])
        if record['parents'] != self.manifest['artifacts'][record['artifact_id']]['parents'] or not record['parents']:
            raise AdmissionError('derived lineage mismatch or empty')
        for measurement in record['measurements']:
            if isinstance(self.measurement(measurement), Absence):
                raise AdmissionError('derived artifact cites unavailable measurement')
        roots = {m['product_id'] for m in record['measurements']}
        if len(roots) != len(record['measurements']) or roots != set(record['parents']):
            raise AdmissionError('derived missing measurement contributions')
        return record


def validate_measurement(record, context):
    return context.measurement(record)


def validate_decisions(rows, schedule, bindings):
    """Exact method/window accounting, including abstentions; bind current fits."""
    expected = {(s['window_id'], m): s for s in schedule for m in bindings}
    if not expected or len({s['window_id'] for s in schedule}) != len(schedule):
        raise AdmissionError('empty/duplicate schedule')
    seen = set()
    for row in rows:
        validate_schema(row, 'decision')
        key = row['window_id'], row['method_id']
        if key not in expected or key in seen:
            raise AdmissionError('unknown/orphan/duplicate decision')
        seen.add(key)
        if row['window_hash'] != digest(expected[key]) or row['binding_hash'] != digest(bindings[row['method_id']]):
            raise AdmissionError('stale window/schema/model/scope binding')
        estimated = row['status'] == 'ESTIMATED'
        if estimated != (row['value'] is not None) or estimated != (row['reason'] is None):
            raise AdmissionError('value/status/reason conflict')
    if seen != set(expected):
        raise AdmissionError('omitted scheduled decision/abstention')
    return {'status': 'INTEGRITY_VALID', 'rows': len(rows), 'research_admissible': False}


def validate_scopes(scopes, membership, exposed_sites):
    """Join independent source/window membership; scopes cannot assign themselves."""
    roles = {'fitting', 'development_validation', 'threshold_calibration', 'evaluation'}
    if set(scopes) != roles:
        raise AdmissionError('four explicit scopes required')
    all_ids, role_sites = set(), {}
    for role, ids in scopes.items():
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise AdmissionError('empty/duplicate ' + role + ' scope')
        role_sites[role] = set()
        for wid in ids:
            if wid in all_ids or wid not in membership or membership[wid]['scope'] != role:
                raise AdmissionError('scope membership conflict')
            all_ids.add(wid)
            row = membership[wid]
            validate_schema(row, 'scope_member')
            if timestamp(row['available_at']) > timestamp(row['cutoff']):
                raise AdmissionError('fit support unavailable at cutoff')
            role_sites[role].add(row['site_id'])
    if all_ids != set(membership):
        raise AdmissionError('omitted scope membership')
    if role_sites['evaluation'] & (set(exposed_sites) | set.union(*(role_sites[r] for r in roles - {'evaluation'}))):
        raise AdmissionError('evaluation site is fitting/calibration/development exposed')
    return {'status': 'INTEGRITY_VALID', 'scopes_hash': digest(scopes), 'membership_hash': digest(membership)}


def fit_fixture_operators(panels, embeddings, calibration_scores, scopes, membership, exposed_sites):
    """Execute analytical fits only after independent scope joins.

    This is an integrity control, not an observational study runner. Window IDs
    and canonical sites come from the independent membership table, not from
    the caller's desired list of fitting lakes.
    """
    import numpy as np
    from .features import FeatureNormalizer, FitScope
    from .scoring import EmbeddingDistanceScorer
    from .calibration import calibrate_false_alert_threshold
    binding = validate_scopes(scopes, membership, exposed_sites)
    fit_ids, cal_ids = scopes['fitting'], scopes['threshold_calibration']
    if set(panels) != set(fit_ids) or set(embeddings) != set(fit_ids) or set(calibration_scores) != set(cal_ids):
        raise AdmissionError('actual fit/calibration inputs disagree with independent membership')
    train_sites = tuple(sorted({membership[w]['site_id'] for w in fit_ids}))
    forbidden = tuple(sorted({membership[w]['site_id'] for w in scopes['evaluation']}))
    arrays, emb = {}, {}
    for wid in fit_ids:
        p = panels[wid]
        if p.window_id != wid:
            raise AdmissionError('actual panel window identity disagrees with membership')
        if p.origin != 'fixture' or p.lake_id != membership[wid]['site_id'] or timestamp(p.end_date + 'T23:59:59+00:00') > timestamp(membership[wid]['cutoff']):
            raise AdmissionError('actual panel site/origin/cutoff disagrees with membership')
        if p.lake_id in arrays:
            raise AdmissionError('duplicate source/site weighting needs a declared policy')
        arrays[p.lake_id] = p.values, p.mask
        emb[p.lake_id] = embeddings[wid]
    scope = FitScope(train_sites, forbidden)
    normalizer = FeatureNormalizer(scope).fit(arrays)
    pca = EmbeddingDistanceScorer(scope, k_neighbors=1, n_components=1).fit(emb)
    threshold = calibrate_false_alert_threshold([calibration_scores[w] for w in cal_ids], cal_ids, scopes['evaluation'], .5)
    return {'normalizer': normalizer, 'pca': pca, 'threshold': threshold, 'binding': binding, 'origin': 'fixture', 'research_admissible': False}


def validate_request(record, context):
    """An unavailable request has no acquisition/support time or physical value."""
    validate_schema(record, 'request')
    if context.manifest.get('requests', {}).get(record['request_id']) != digest(record):
        raise AdmissionError('missing or mismatched request membership')
    if record['origin'] != context.mode:
        raise AdmissionError('request origin mismatch')
    timestamp(record['requested_at'])
    query, response = context.json(record['query_id']), context.json(record['response_id'])
    if query.get('endpoint') != record['endpoint'] or response.get('query_id') != record['query_id'] or response.get('status') != record['status']:
        raise AdmissionError('request query/response linkage mismatch')
    return Absence('NOT_MEASURED', record['reason'])
