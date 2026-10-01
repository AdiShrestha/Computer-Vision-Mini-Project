"""Read/hash every current file; index every text line without running producers.

This is coverage evidence, not automatic scientific validation. Historical
materials and constructed fixtures are explicitly classified and never used
as measurements. No pickle/checkpoint/archive deserialization occurs.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re

EXCLUDE_DIRS = {'.git', '.migration', '__pycache__', '.pytest_cache', '.venv', '.credentials', '.factory'}
SIGNALS = re.compile(r'fabricat|hardcod|fallback|random|nan_to_num|TODO|REPLACE_ME|SEALED|HMAC|threshold|^\s*(def |class |assert |except )', re.I)


def role(relative):
    if relative.startswith('source/sentinel_gl/'):
        return 'active_numerical_core'
    if relative.startswith(('source/tests/', 'factory/tests/')):
        return 'constructed_engineering_fixtures'
    if relative.startswith('factory/legacy/'):
        return 'historical_factory_not_active'
    if relative.startswith(('docs/audit/', 'docs/audit_tools/')):
        return 'audit_context_not_measurement'
    if relative.startswith('docs/research/deep_research_supplied') or relative in {'docs/research/claude_review_1.txt', 'docs/research/claude_review_2.txt'}:
        return 'untrusted_user_supplied_research_context'
    if relative.startswith('factory/'):
        return 'active_factory_policy_code_or_documentation'
    if relative in {'docs/FORENSIC_AUDIT.md', 'docs/SCHEMA.md', 'docs/VALIDATION.md', 'docs/RELEASE_SCOPE.md', 'docs/REFERENCES.md', 'docs/v26_forensic_results.json', 'docs/COVERAGE.md'}:
        return 'supplied_factory_background_not_sentinel_measurement'
    return 'project_contract_documentation_or_configuration'


def inspect(root, output):
    rows = []
    for directory, subdirs, names in os.walk(root, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs if d not in EXCLUDE_DIRS)
        # Directory symlinks are recorded, never descended.
        for name in list(subdirs):
            path = Path(directory)/name
            if path.is_symlink():
                subdirs.remove(name)
                rows.append({'path': path.relative_to(root).as_posix(), 'inspection': 'symlink_not_followed'})
        for name in sorted(names):
            path = Path(directory)/name
            if path == output or name == '.DS_Store':
                continue
            relative = path.relative_to(root).as_posix()
            row = {'path': relative, 'role': role(relative)}
            if path.is_symlink():
                row['inspection'] = 'symlink_not_followed'
            elif not path.is_file():
                row['inspection'] = 'special_file_not_read'
            else:
                raw = path.read_bytes()
                row.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                try:
                    if b'\0' in raw:
                        raise UnicodeError('binary content')
                    value = raw.decode('utf-8')
                    lines = value.splitlines()
                    row.update(inspection='all_text_lines_read_and_indexed', line_count=len(lines))
                    row['review_signals'] = [{'line': i, 'text': text[:240]} for i, text in enumerate(lines, 1) if SIGNALS.search(text)]
                    if path.suffix == '.py':
                        try:
                            tree = ast.parse(value, filename=relative)
                            row['ast_valid'] = True
                            row['definitions'] = [{'name': node.name, 'start': node.lineno, 'end': node.end_lineno} for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                        except SyntaxError as error:
                            row.update(ast_valid=False, parse_error=str(error))
                except UnicodeError:
                    row['inspection'] = 'binary_bytes_hashed_not_executed'
            rows.append(row)
    rows.sort(key=lambda row: row['path'])
    return {'kind': 'current_checkout_inspection_not_research_validation',
        'scope': 'every nonexcluded current file; every UTF-8 text line indexed; Python AST checked; no unsafe binary execution',
        'excluded_directories': sorted(EXCLUDE_DIRS), 'excluded_files': ['.DS_Store', str(output.relative_to(root))],
        'semantic_review_limit': 'all active numerical-core operators and safety-critical factory execution/evidence paths reviewed; text reading, hashing and AST validity do not prove every scientific or policy claim',
        'file_count': len(rows), 'text_line_count': sum(row.get('line_count', 0) for row in rows), 'files': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inspect(args.root.resolve(), args.output.resolve()), indent=2, allow_nan=False)+'\n')
