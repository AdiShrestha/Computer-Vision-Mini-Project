"""Verify exact distributed file membership and SHA-256 bytes offline."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    rows = json.loads((ROOT / 'FILE_MEMBERSHIP.json').read_text())['files']
    paths = [row['path'] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate distribution paths')
    for row in rows:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe distribution path')
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or non-regular distribution file: ' + row['path'])
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Changed distribution bytes: ' + row['path'])
    sums = {}
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        sha, path = line.split('  ', 1)
        if path in sums:
            raise ValueError('Duplicate checksum path')
        sums[path] = sha
    expected = {row['path']: row['sha256'] for row in rows}
    expected['FILE_MEMBERSHIP.json'] = digest(ROOT / 'FILE_MEMBERSHIP.json')
    if sums != expected:
        raise ValueError('Checksum ledger and membership manifest disagree')
    print(json.dumps({'status': 'PASS', 'payload_files': len(rows),
                      'checksums_verified': len(sums)}, sort_keys=True))


if __name__ == '__main__':
    sys.exit(main())
