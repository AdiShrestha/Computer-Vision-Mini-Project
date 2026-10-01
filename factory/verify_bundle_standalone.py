#!/usr/bin/env python3
"""Standalone bundle verifier. Does NOT import gatekeeper or any engine module.

This is the independent verifier specified by the trust model: it must not use
the project's own gatekeeper.py to verify the gatekeeper's claims. It verifies:

  - ZIP membership and byte integrity
  - Bundle manifest schema
  - Receipt signatures (when keys are available)
  - Assurance level consistency

Usage:
    python3 verify_bundle_standalone.py <archive.zip> [--public-key <key.pub>]
"""
import argparse
import base64
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = 'BUNDLE_MANIFEST.json'


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: '+key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON constant: '+value)
    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite JSON number: '+value)
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject, parse_float=finite)


def _verify_receipt_sig(receipt_data, pub_key_bytes, scheme):
    if not isinstance(receipt_data, dict):
        return False, "receipt is not a JSON object"
    sig_b64 = receipt_data.get('supervisor_signature')
    rec_scheme = receipt_data.get('signature_scheme')
    if not sig_b64 or not rec_scheme:
        return False, "receipt missing supervisor signature or scheme"
    if rec_scheme != scheme:
        return False, f"signature scheme mismatch: receipt has {rec_scheme}, key is {scheme}"
    try:
        sig = base64.b64decode(sig_b64, validate=True)
    except Exception:
        return False, "malformed base64 signature"
    to_verify = {k: v for k, v in receipt_data.items()
                 if k not in ('supervisor_signature', 'signature_scheme', 'public_key_id')}
    payload = canonical(to_verify)
    if receipt_data.get('public_key_id') != hashlib.sha256(pub_key_bytes).hexdigest()[:16]:
        return False, 'receipt public-key identity mismatch'
    if scheme == 'ed25519':
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            pub = Ed25519PublicKey.from_public_bytes(pub_key_bytes)
            pub.verify(sig, payload)
            return True, None
        except Exception as e:
            return False, f"ed25519 signature verification failed: {e}"
    else:
        return False, f"unsupported signature scheme: {scheme}"


def verify_bundle(path, public_key_path=None):
    """Verify bundle integrity without importing any engine module."""
    errors = []
    path = Path(path)

    if not path.is_file():
        return {'status': 'FAIL', 'errors': ['archive not found']}

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]

            # Check for unsafe names
            for name in names:
                member = PurePosixPath(name)
                if not name or '..' in member.parts or member.is_absolute() or '\\' in name or ':' in name or str(member) != name or name == '.':
                    errors.append(f'unsafe member name: {name}')

            # Duplicates
            if len(names) != len(set(names)):
                errors.append('duplicate members in archive')

            # Manifest present
            if MANIFEST not in names:
                errors.append('missing bundle manifest')
                return {'status': 'FAIL', 'errors': errors}

            # No symlinks or directories
            for info in infos:
                if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    errors.append(f'bundle contains non-regular file: {info.filename}')

            # Size limit on manifest
            manifest_info = archive.getinfo(MANIFEST)
            if manifest_info.file_size > 16 * 1024 * 1024:
                errors.append('manifest exceeds 16 MiB limit')
                return {'status': 'FAIL', 'errors': errors}

            # Parse manifest
            raw = archive.read(MANIFEST)
            try:
                manifest = strict_json(raw)
            except ValueError as e:
                errors.append(f'invalid manifest JSON: {e}')
                return {'status': 'FAIL', 'errors': errors}

            if not isinstance(manifest, dict):
                errors.append('manifest must be an object')
                return {'status': 'FAIL', 'errors': errors}

            if manifest.get('schema_version') != 1:
                errors.append(f'unsupported schema_version: {manifest.get("schema_version")}')

            files = manifest.get('files')
            if not isinstance(files, dict):
                errors.append('manifest.files must be an object')
                return {'status': 'FAIL', 'errors': errors}

            # Membership check
            archive_members = set(names) - {MANIFEST}
            manifest_members = set(files.keys())
            extra_in_archive = archive_members - manifest_members
            extra_in_manifest = manifest_members - archive_members
            if extra_in_archive:
                errors.append(f'archive has unlisted members: {sorted(extra_in_archive)}')
            if extra_in_manifest:
                errors.append(f'manifest lists missing members: {sorted(extra_in_manifest)}')

            # Byte integrity
            for name, expected_hash in files.items():
                if name not in archive_members:
                    continue
                h = hashlib.sha256()
                with archive.open(name) as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(chunk)
                if h.hexdigest() != expected_hash:
                    errors.append(f'hash mismatch: {name}')

            signatures_verified = 0
            execution_bindings_verified = 0
            execution_members = sorted(name for name in archive_members if name.endswith('execution.json'))
            if public_key_path and not errors:
                pk_path = Path(public_key_path)
                if not pk_path.is_file():
                    errors.append(f'public key file not found: {public_key_path}')
                else:
                    try:
                        lines = pk_path.read_text().splitlines()
                        if len(lines) != 2 or lines[0].strip() != '# ed25519':
                            raise ValueError('only Ed25519 public verification keys are supported')
                        scheme = 'ed25519'
                        pub_bytes = base64.b64decode(lines[1], validate=True)
                        if len(pub_bytes) != 32:
                            raise ValueError('malformed Ed25519 public key')

                        for name in archive_members:
                            if name.endswith('execution.json'):
                                try:
                                    record = strict_json(archive.read(name))
                                    if not isinstance(record, dict):
                                        raise ValueError('execution record must be an object')
                                    receipt_json = record.get('supervisor_receipt', record)
                                    ok, msg = _verify_receipt_sig(receipt_json, pub_bytes, scheme)
                                    if not ok:
                                        errors.append(f'invalid receipt signature in {name}: {msg}')
                                    else:
                                        signatures_verified += 1
                                        if 'supervisor_receipt' in record:
                                            bound = {key: record.get(key) for key in ('run_nonce', 'epoch', 'experiment_id', 'seed', 'started_at', 'finished_at', 'interpreter_hash', 'dependency_lock_hash', 'snapshot_merkle_root')}
                                            bound.update({'exit_status': record.get('exit_code'),
                                                'launch_spec': sha256_bytes(canonical(record.get('argv'))),
                                                'input_root': sha256_bytes(canonical(record.get('inputs_before'))),
                                                'output_root': sha256_bytes(canonical(record.get('outputs')))})
                                            if any(type(receipt_json.get(k)) is not type(v) or receipt_json.get(k) != v for k,v in bound.items()):
                                                raise ValueError('signed fields disagree with execution record')
                                            if record.get('inputs_before') != record.get('inputs_after'):
                                                raise ValueError('execution inputs changed')
                                            outputs = record.get('outputs')
                                            if not isinstance(outputs, dict) or not outputs or any(files.get(k) != v for k,v in outputs.items()):
                                                raise ValueError('execution output membership/hashes are not present in bundle')
                                            execution_bindings_verified += 1
                                except Exception as e:
                                    errors.append(f'could not verify receipt in {name}: {e}')
                    except Exception as e:
                        errors.append(f'failed to load public key: {e}')

    except (zipfile.BadZipFile, OSError) as e:
        return {'status': 'FAIL', 'errors': [f'cannot open archive: {e}']}

    result = {
        'status': 'PASS' if not errors else 'FAIL',
        'files_checked': len(files) if not errors else 0,
        'signatures_verified': signatures_verified if not errors else 0,
        'execution_bindings_verified': execution_bindings_verified if not errors else 0,
        'errors': errors,
        'release_status': manifest.get('release_status', 'unknown'),
        'factory_version': manifest.get('factory_version', 'unknown'),
        'declared_assurance_level': manifest.get('assurance_level', 'unknown'),
        'assurance_level': 'BLOCKED' if errors else ('SUPERVISOR_ATTESTED' if execution_members and execution_bindings_verified == len(execution_members) else 'STRUCTURALLY_VALIDATED'),
        'scope': 'standalone membership, bytes, optional trusted Ed25519 signatures and local execution-field/output bindings; no sealed isolation, provider authenticity, or domain/scientific validation',
    }

    return result



def main():
    parser = argparse.ArgumentParser(description='Standalone bundle verifier')
    parser.add_argument('archive', help='Path to the bundle ZIP')
    parser.add_argument('--public-key', help='Path to supervisor public key')
    args = parser.parse_args()

    result = verify_bundle(args.archive, args.public_key)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
