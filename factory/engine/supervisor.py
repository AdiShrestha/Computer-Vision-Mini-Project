"""Local receipt signatures and runtime metadata with an explicit trust limit.

Ed25519 is required. Verification reads only the public key. No HMAC secret
is exported as public verification material. Workers still run as the same OS
user, so signatures do not establish a hostile-worker isolation boundary.
"""
import base64
import hashlib
import locale
import os
import platform
import sys
import time
from pathlib import Path

from .io import canonical, sha, EvidenceError

# Key storage location — outside any project workspace
_DEFAULT_KEY_DIR = Path.home() / '.factory'
_KEY_ENV = 'FACTORY_SUPERVISOR_KEY'

# Signature schemes
SCHEME_ED25519 = 'ed25519'


def _key_dir():
    """Resolve the supervisor key directory."""
    env = os.environ.get(_KEY_ENV)
    if env:
        return Path(env).parent
    return _DEFAULT_KEY_DIR


def _key_path():
    """Resolve the path to the supervisor private key."""
    env = os.environ.get(_KEY_ENV)
    if env:
        return Path(env)
    return _DEFAULT_KEY_DIR / 'supervisor.key'


def _pub_key_path():
    """Resolve the path to the supervisor public key."""
    return _key_path().with_suffix('.pub')


def _try_ed25519():
    """Check if real Ed25519 is available."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        return True
    except ImportError:
        return False


def init_supervisor_keys(force=False):
    """Generate supervisor keypair if it doesn't exist.

    Returns (private_key_path, public_key_path, scheme).
    """
    priv = _key_path()
    pub = _pub_key_path()

    if not _try_ed25519():
        raise EvidenceError('Ed25519 signing requires cryptography; HMAC public-key fallback is forbidden')
    if (priv.exists() or pub.exists()) and not force:
        if not (priv.exists() and pub.exists()):
            raise EvidenceError('incomplete supervisor keypair; explicit repair is required')
        _, scheme = _load_public_key(pub)
        return priv, pub, scheme

    priv.parent.mkdir(parents=True, exist_ok=True)

    if _try_ed25519():
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization

        private_key = Ed25519PrivateKey.generate()
        priv_bytes = private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption()
        )
        pub_bytes = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw
        )
        priv.write_bytes(priv_bytes)
        os.chmod(priv, 0o600)
        pub.write_text(f'# {SCHEME_ED25519}\n{base64.b64encode(pub_bytes).decode()}\n')
        return priv, pub, SCHEME_ED25519


def _load_public_key(path=None):
    pub = Path(path) if path is not None else _pub_key_path()
    try:
        lines = pub.read_text().splitlines()
        if len(lines) != 2 or lines[0].strip() != '# ed25519':
            raise EvidenceError('only Ed25519 public verification is supported; legacy HMAC keys require explicit rotation')
        data = base64.b64decode(lines[1], validate=True)
        if len(data) != 32:
            raise EvidenceError('malformed Ed25519 public key')
    except (OSError, ValueError) as error:
        raise EvidenceError('public verification key unavailable or malformed') from error
    return data, SCHEME_ED25519


def _load_keys():
    priv, pub = _key_path(), _pub_key_path()
    if not priv.exists() and not pub.exists():
        init_supervisor_keys()
    pub_data, scheme = _load_public_key(pub)
    if not _try_ed25519():
        raise EvidenceError('Ed25519 signing requires cryptography')
    try:
        priv_bytes = priv.read_bytes()
    except OSError as error:
        raise EvidenceError('private signing key unavailable') from error
    if len(priv_bytes) != 32:
        raise EvidenceError('malformed private signing key')
    return priv_bytes, pub_data, scheme


def sign_receipt(receipt_dict):
    """Sign a receipt dictionary. Returns the receipt with 'supervisor_signature' added.

    The receipt is canonicalized (sorted keys, compact JSON) before signing.
    The signature covers the canonical bytes.
    """
    priv_bytes, pub_data, scheme = _load_keys()

    # Remove any prior signature before signing
    to_sign = {k: v for k, v in receipt_dict.items()
               if k not in ('supervisor_signature', 'signature_scheme', 'public_key_id')}
    payload = canonical(to_sign)

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
    sig_b64 = base64.b64encode(private_key.sign(payload)).decode()

    pub_id = hashlib.sha256(pub_data).hexdigest()[:16]

    signed = dict(receipt_dict)
    signed['supervisor_signature'] = sig_b64
    signed['signature_scheme'] = scheme
    signed['public_key_id'] = pub_id

    return signed


def verify_receipt_signature(receipt_dict, *, public_key_path=None):
    """Verify the supervisor signature on a receipt.

    Returns True if valid, raises EvidenceError if invalid or missing.
    """
    sig_b64 = receipt_dict.get('supervisor_signature')
    scheme = receipt_dict.get('signature_scheme')
    key_id = receipt_dict.get('public_key_id')

    if not sig_b64 or not scheme:
        raise EvidenceError('receipt is missing supervisor signature')

    to_verify = {k: v for k, v in receipt_dict.items()
                 if k not in ('supervisor_signature', 'signature_scheme', 'public_key_id')}
    payload = canonical(to_verify)

    pub_data, stored_scheme = _load_public_key(public_key_path)
    if scheme != stored_scheme or scheme != SCHEME_ED25519:
        raise EvidenceError("receipt signature scheme is not the trusted Ed25519 scheme")

    # Verify key identity
    expected_id = hashlib.sha256(pub_data).hexdigest()[:16]
    if key_id != expected_id:
        raise EvidenceError('receipt signed by unknown supervisor key')

    try:
        sig = base64.b64decode(sig_b64, validate=True)
    except Exception:
        raise EvidenceError('receipt signature is malformed or truncated')

    if scheme == SCHEME_ED25519 and _try_ed25519():
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        public_key = Ed25519PublicKey.from_public_bytes(pub_data)
        try:
            public_key.verify(sig, payload)
        except Exception:
            raise EvidenceError('receipt signature verification failed')
    else:
        raise EvidenceError(f'unknown signature scheme: {scheme}')

    return True


def build_receipt(*, run_nonce, project_id, epoch, experiment_id,
                  snapshot_merkle_root, input_root, runtime_id,
                  interpreter_hash, dependency_lock_hash, launch_spec,
                  seed, output_root, exit_status, cpu_time, memory_peak,
                  started_at, finished_at, supervisor_version, policy_version,
                  resource_observations=None):
    """Build a complete supervisor receipt binding all 16 required fields."""
    receipt = {
        'receipt_version': 2,
        'run_nonce': run_nonce,
        'project_id': project_id,
        'epoch': epoch,
        'experiment_id': experiment_id,
        'snapshot_merkle_root': snapshot_merkle_root,
        'input_root': input_root,
        'runtime_id': runtime_id,
        'interpreter_hash': interpreter_hash,
        'dependency_lock_hash': dependency_lock_hash,
        'launch_spec': launch_spec,
        'seed': seed,
        'output_root': output_root,
        'exit_status': exit_status,
        'resource_observations': resource_observations if resource_observations is not None else {
            'cpu_time_seconds': cpu_time, 'memory_peak_bytes': memory_peak,
            'measurement_method': 'caller_supplied_not_independently_verified',
        },
        'started_at': started_at,
        'finished_at': finished_at,
        'supervisor_version': supervisor_version,
        'policy_version': policy_version,
    }
    return sign_receipt(receipt)


def runtime_attestation():
    """Capture current runtime environment for binding into receipts."""
    try:
        loc = locale.getlocale()
    except Exception:
        loc = ('unknown', 'unknown')

    return {
        'interpreter_binary': sys.executable,
        'interpreter_hash': _interpreter_hash(),
        'python_version': sys.version,
        'platform_system': platform.system(),
        'platform_release': platform.release(),
        'platform_machine': platform.machine(),
        'platform_node': platform.node(),
        'locale': str(loc),
        'timezone': str(time.timezone),
        'encoding': sys.getdefaultencoding(),
        'byte_order': sys.byteorder,
    }


def _interpreter_hash():
    """SHA-256 of the running interpreter binary."""
    h = hashlib.sha256()
    try:
        with open(sys.executable, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
    except OSError:
        return 'unavailable'
    return h.hexdigest()
