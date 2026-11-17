"""Versioned task contracts and independent row-level integrity recomputation.

These checks establish structural integrity only. A valid fixture or a
well-formed observational record is not a scientific result or an admitted
measurement; observational decoding remains separately blocked.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from .admission import AdmissionError, schema_check
from .integrity import digest

SCHEMA_DIR = Path(__file__).with_name('schemas') / 'tasks'
SCOPE_ROLES = frozenset({'fitting', 'development_validation', 'threshold_calibration', 'evaluation'})
ORIGINS = frozenset({'observational', 'fixture', 'simulation'})
_SHA256 = re.compile(r'[0-9a-f]{64}')
_YEAR_DAYS = 365.2425


def _fail(message):
    raise AdmissionError(message)


def _time(value, *, date_only=False):
    if not isinstance(value, str):
        _fail('timestamp must be a string')
    try:
        if date_only:
            result = datetime.fromisoformat(value + 'T00:00:00+00:00')
        else:
            result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise AdmissionError('invalid timestamp') from error
    if not date_only and result.tzinfo is None:
        _fail('timestamp timezone required')
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def _keys(row, required, optional=()):
    if not isinstance(row, dict) or not set(required) <= row.keys() or row.keys() - set(required) - set(optional):
        _fail('row fields do not match versioned task contract')


def _ids(value, *, allow_empty=False):
    if not isinstance(value, list) or (not allow_empty and not value):
        _fail('membership must be an ordered ID list')
    if any(not isinstance(item, str) or not item.strip() for item in value) or len(set(value)) != len(value):
        _fail('membership has an empty or duplicate ID')
    return set(value)


def _finite_or_null(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value))


def _table(table, key):
    if isinstance(table, dict):
        rows = table
    elif isinstance(table, list):
        rows = {}
        for row in table:
            if not isinstance(row, dict) or not isinstance(row.get(key), str) or row[key] in rows:
                _fail('duplicate or malformed ' + key + ' table')
            rows[row[key]] = row
    else:
        _fail('source and membership tables must be mappings or row lists')
    return rows


def _source_references(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'source_ids', 'parent_source_ids'}:
                found.update(_ids(item, allow_empty=True))
            else:
                _source_references(item, found)
    elif isinstance(value, list):
        for item in value:
            _source_references(item, found)
    return found


def _validate_inputs(task, source_rows, scope_membership):
    sources = _table(source_rows, 'source_id')
    scopes = _table(scope_membership, 'source_id')
    declared = _ids(task['source_ids'])
    referenced = _source_references({k: v for k, v in task.items() if k != 'source_ids'})
    if referenced != declared:
        _fail('task source_ids do not exactly join to row source references')
    if not declared <= sources.keys() or not declared <= scopes.keys():
        _fail('task contains an orphan source or scope membership')
    for source_id in declared:
        row = sources[source_id]
        _keys(row, ('source_id', 'sha256', 'origin', 'status'), ('support_start', 'support_end', 'available_at', 'reason'))
        if row['source_id'] != source_id or not isinstance(row['sha256'], str) or not _SHA256.fullmatch(row['sha256']):
            _fail('source row identity or SHA-256 is invalid')
        if row['origin'] not in ORIGINS or row['status'] not in {'MEASURED', 'NO_DATA', 'FAILED', 'UNKNOWN', 'DERIVED'}:
            _fail('unknown source origin or status')
        if row['status'] == 'MEASURED':
            for field in ('support_start', 'support_end', 'available_at'):
                if field not in row or row[field] is None:
                    _fail('measured source lacks closed-support timestamps')
                _time(row[field])
            if _time(row['support_end']) < _time(row['support_start']) or _time(row['available_at']) < _time(row['support_end']):
                _fail('measured source support or availability is not closed')
            if row.get('reason') is not None:
                _fail('measured source cannot carry an absence reason')
        elif not isinstance(row.get('reason'), str) or not row['reason'].strip():
            _fail('missing/failed/derived source requires an explicit reason')
        scope = scopes[source_id]
        _keys(scope, ('source_id', 'scope_role', 'physical_status', 'physical_unit_id'), ('evaluation_subgroup',))
        if scope['source_id'] != source_id or scope['scope_role'] not in SCOPE_ROLES:
            _fail('source scope membership is malformed')
        if scope['physical_status'] == 'KNOWN':
            if not isinstance(scope['physical_unit_id'], str) or not scope['physical_unit_id'].strip():
                _fail('known physical membership requires its source-assigned unit ID')
        elif scope['physical_status'] == 'UNKNOWN':
            if scope['physical_unit_id'] is not None:
                _fail('unknown physical membership cannot carry a unit ID')
        else:
            _fail('physical membership status must be KNOWN or UNKNOWN')
        if task['task_type'] == 'SIMULATION':
            if scope['scope_role'] != 'development_validation' or row['origin'] == 'observational':
                _fail('simulation lineage is restricted to nonobservational development inputs')
        elif row['origin'] != task['origin']:
            _fail('source origin does not match task origin')
        if scope['scope_role'] != task['scope_role']:
            _fail('source scope role does not match task scope')
        if task['scope_role'] == 'evaluation' and 'evaluation_subgroup' in scope:
            if not isinstance(scope['evaluation_subgroup'], str) or not scope['evaluation_subgroup'].strip():
                _fail('evaluation subgroup must be explicit when present')
        elif 'evaluation_subgroup' in scope:
            _fail('evaluation subgroups cannot redefine a non-evaluation scope')
    return sources, scopes


def _validate_schedule(schedule, methods, sources, scopes, *, decisions, decision_kind, expected_schedule):
    if not isinstance(schedule, list) or not schedule:
        _fail('complete nonempty schedule required')
    if not isinstance(expected_schedule, list) or not expected_schedule:
        _fail('independently supplied complete schedule required')
    canonical = lambda rows: sorted((json.dumps(row, sort_keys=True, separators=(',', ':')) for row in rows))
    if canonical(schedule) != canonical(expected_schedule):
        _fail('task schedule does not exactly match the independent schedule rows')
    by_window = {}
    for row in schedule:
        _keys(row, ('window_id', 'source_ids', 'support_start', 'support_end', 'decision_at', 'eligibility_status', 'reason'))
        wid = row['window_id']
        if not isinstance(wid, str) or not wid.strip() or wid in by_window:
            _fail('empty or duplicate scheduled window')
        sources_for_window = _ids(row['source_ids'], allow_empty=True)
        if not sources_for_window <= sources.keys():
            _fail('schedule refers to an unknown source')
        start, end, decision = (_time(row[k]) for k in ('support_start', 'support_end', 'decision_at'))
        if end < start or end > decision:
            _fail('schedule support is not closed by its decision time')
        if row['eligibility_status'] not in {'ELIGIBLE', 'INELIGIBLE'}:
            _fail('unknown eligibility status')
        if row['eligibility_status'] == 'ELIGIBLE' and row['reason'] is not None:
            _fail('eligible schedule row cannot carry a missingness reason')
        if row['eligibility_status'] == 'INELIGIBLE' and (not isinstance(row['reason'], str) or not row['reason'].strip()):
            _fail('ineligible schedule row requires an explicit reason')
        for source_id in sources_for_window:
            source = sources[source_id]
            if source['status'] == 'MEASURED' and _time(source['available_at']) > decision:
                _fail('measurement was unavailable at the declared decision time')
            if source['status'] == 'MEASURED' and (_time(source['support_start']) < start or _time(source['support_end']) > end):
                _fail('measurement support falls outside its scheduled window')
        by_window[wid] = row
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or any(not isinstance(x, str) or not x.strip() for x in methods):
        _fail('unique explicit method IDs required')
    expected = {(wid, method) for wid in by_window for method in methods}
    seen = set()
    for row in decisions:
        if not isinstance(row, dict) or not isinstance(row.get('window_id'), str) or not isinstance(row.get('method_id'), str):
            _fail('malformed scheduled task row')
        key = row['window_id'], row['method_id']
        if key not in expected or key in seen:
            _fail('orphan or duplicate scheduled task row')
        if decision_kind == 'observability':
            _keys(row, ('window_id', 'method_id', 'source_ids', 'status', 'value', 'reason'))
            states = {'ESTIMATED', 'ABSTAINED', 'NOT_ESTIMABLE'}
            estimated = row['status'] == 'ESTIMATED'
            if row['status'] not in states or not _finite_or_null(row['value']):
                _fail('invalid observability value or status')
            if estimated != (row['value'] is not None) or estimated != (row['reason'] is None):
                _fail('observability null/status/reason mismatch')
            if not estimated and (not isinstance(row['reason'], str) or not row['reason'].strip()):
                _fail('abstention and not-estimable rows require reasons')
        else:
            _keys(row, ('window_id', 'method_id', 'source_ids', 'status', 'declaration_at', 'reason'))
            if row['status'] not in {'ALARM_DECLARED', 'NO_ALARM', 'ABSTAINED', 'NOT_ESTIMABLE'}:
                _fail('invalid alarm declaration status')
            declared = row['status'] == 'ALARM_DECLARED'
            if declared:
                if row['declaration_at'] is None or row['reason'] is not None:
                    _fail('alarm declaration requires a time and no absence reason')
                _time(row['declaration_at'])
                if _time(row['declaration_at']) != _time(by_window[row['window_id']]['decision_at']):
                    _fail('alarm declaration must equal its actual scheduled decision arrival')
            elif row['declaration_at'] is not None:
                _fail('nonalarm/abstention cannot carry an alarm declaration time')
            elif row['status'] == 'NO_ALARM' and row['reason'] is not None:
                _fail('no-alarm row cannot carry a missingness reason')
            elif row['status'] in {'ABSTAINED', 'NOT_ESTIMABLE'} and (not isinstance(row['reason'], str) or not row['reason'].strip()):
                _fail('abstention and not-estimable alarm rows require reasons')
        seen.add(key)
        if _ids(row['source_ids'], allow_empty=True) != _ids(by_window[key[0]]['source_ids'], allow_empty=True):
            _fail('task row sources do not match the scheduled window sources')
        if by_window[key[0]]['eligibility_status'] == 'INELIGIBLE' and row['status'] in {'ESTIMATED', 'ALARM_DECLARED', 'NO_ALARM'}:
            _fail('ineligible schedule cannot produce an estimated decision')
    if seen != expected:
        _fail('omitted scheduled decision, method or explicit abstention')
    return by_window


def _same_physical_unit(source_ids, scopes, expected):
    units = {scopes[source_id]['physical_unit_id'] for source_id in source_ids}
    statuses = {scopes[source_id]['physical_status'] for source_id in source_ids}
    if expected is None:
        if statuses != {'UNKNOWN'} or units != {None}:
            _fail('unknown physical unit disagrees with source membership')
    elif statuses != {'KNOWN'} or units != {expected}:
        _fail('physical unit does not join to source membership')


def validate_sampling_provenance_graph(graph, source_rows, scope_membership):
    """Check declared graph structure; declarations alone cannot establish independent N."""
    digest(graph)
    digest(source_rows)
    digest(scope_membership)
    _keys(graph, ('G_S', 'G_P'))
    _keys(graph['G_S'], ('nodes', 'edges'))
    _keys(graph['G_P'], ('nodes', 'edges'))
    sources = _table(source_rows, 'source_id')
    scopes = _table(scope_membership, 'source_id')
    physical_nodes = {}
    for node in graph['G_S']['nodes']:
        _keys(node, ('node_id', 'source_ids', 'physical_status', 'physical_unit_id'))
        node_id = node['node_id']
        if not isinstance(node_id, str) or not node_id.strip() or node_id in physical_nodes:
            _fail('duplicate physical sampling node')
        ids = _ids(node['source_ids'])
        if not ids <= sources.keys() or not ids <= scopes.keys():
            _fail('physical node has an unknown source membership')
        _same_physical_unit(ids, scopes, node['physical_unit_id'] if node['physical_status'] == 'KNOWN' else None)
        if any(scopes[source_id]['physical_status'] != node['physical_status'] for source_id in ids):
            _fail('physical node status conflicts with source membership')
        physical_nodes[node_id] = node
    graph_sources = {source_id for node in physical_nodes.values() for source_id in node['source_ids']}
    if graph_sources != set(scopes):
        _fail('G_S must preserve every independently supplied physical membership row')
    allowed_physical = {'same_physical_unit', 'overlapping_geographic_support', 'shared_waterbody',
                        'shared_glacier', 'shared_river_basin', 'shared_era5_cell',
                        'overlapping_sentinel_frame', 'proximity_lt_50km', 'shared_event', 'overlapping_temporal_support'}
    physical_adjacency = {}
    for node in physical_nodes.values():
        if node['physical_status'] == 'KNOWN':
            physical_adjacency.setdefault(node['physical_unit_id'], set())
    for edge in graph['G_S']['edges']:
        _keys(edge, ('source', 'target', 'relation', 'evidence_source_ids'))
        if edge['source'] not in physical_nodes or edge['target'] not in physical_nodes or edge['source'] == edge['target']:
            _fail('physical edge endpoint is unknown or self-referential')
        if edge['relation'] not in allowed_physical:
            _fail('provenance relation cannot be placed in G_S')
        evidence = _ids(edge['evidence_source_ids'])
        if not evidence <= sources.keys() or any(sources[sid]['origin'] == 'simulation' for sid in evidence):
            _fail('physical edge lacks nonsimulated source evidence')
        if physical_nodes[edge['source']]['physical_status'] != 'KNOWN' or physical_nodes[edge['target']]['physical_status'] != 'KNOWN':
            _fail('unknown physical membership cannot be resolved by an inferred edge')
        left = physical_nodes[edge['source']]['physical_unit_id']
        right = physical_nodes[edge['target']]['physical_unit_id']
        if edge['relation'] == 'same_physical_unit' and left != right:
            _fail('same-physical-unit edge conflicts with source membership')
        if not evidence <= set(physical_nodes[edge['source']]['source_ids']) | set(physical_nodes[edge['target']]['source_ids']):
            _fail('physical edge evidence is not joined to its endpoints')
        physical_adjacency[left].add(right)
        physical_adjacency[right].add(left)
    provenance_nodes = {}
    for node in graph['G_P']['nodes']:
        _keys(node, ('node_id', 'kind', 'source_ids'))
        if node['kind'] not in {'run', 'seed', 'checkpoint', 'transform', 'input'}:
            _fail('unknown provenance node type')
        if not isinstance(node['node_id'], str) or not node['node_id'].strip() or node['node_id'] in provenance_nodes:
            _fail('duplicate provenance node')
        ids = _ids(node['source_ids'], allow_empty=True)
        if not ids <= sources.keys():
            _fail('provenance node has an unknown source')
        provenance_nodes[node['node_id']] = node
    allowed_provenance = {'derived_from', 'trained_from', 'checkpoint_of', 'seed_for', 'input_to'}
    provenance_adjacency = {node: [] for node in provenance_nodes}
    for edge in graph['G_P']['edges']:
        _keys(edge, ('source', 'target', 'relation'))
        if edge['source'] not in provenance_nodes or edge['target'] not in provenance_nodes or edge['source'] == edge['target']:
            _fail('provenance edge endpoint is unknown or self-referential')
        if edge['relation'] not in allowed_provenance:
            _fail('physical sampling relation cannot be placed in G_P')
        provenance_adjacency[edge['source']].append(edge['target'])
    active, done = set(), set()
    def visit(node):
        if node in active:
            _fail('provenance graph must be acyclic')
        if node in done:
            return
        active.add(node)
        for child in provenance_adjacency[node]:
            visit(child)
        active.remove(node)
        done.add(node)
    for node in provenance_nodes:
        visit(node)
    known = set(physical_adjacency)
    components, visited = [], set()
    for unit_id in sorted(known):
        if unit_id in visited:
            continue
        component, queue = [], [unit_id]
        visited.add(unit_id)
        while queue:
            current = queue.pop(0)
            component.append(current)
            for adjacent in sorted(physical_adjacency[current]):
                if adjacent not in visited:
                    visited.add(adjacent)
                    queue.append(adjacent)
        components.append(sorted(component))
    unknown = sorted(sid for sid, row in scopes.items() if row['physical_status'] == 'UNKNOWN')
    return {'status': 'INTEGRITY_VALID', 'known_physical_units': len(known),
            'physical_unit_ids': sorted(known), 'declared_sampling_clusters': components,
            'declared_sampling_n': len(components), 'unresolved_source_ids': unknown,
            'independent_sampling_n': None, 'independence_status': 'NOT_ESTABLISHED',
            'independence_reason': 'DECLARED_GRAPH_NOT_AUTHENTICATED_PHYSICAL_ADMISSION',
            'seed_count': sum(node['kind'] == 'seed' for node in provenance_nodes.values()),
            'research_admissible': False}


def _observability(task, sources, scopes):
    windows = _validate_schedule(task['schedule'], task['method_ids'], sources, scopes,
                                 decisions=task['decisions'], decision_kind='observability',
                                 expected_schedule=task['_expected_schedule'])
    estimated = sum(row['status'] == 'ESTIMATED' for row in task['decisions'])
    total = len(task['decisions'])
    eligible = sum(row['eligibility_status'] == 'ELIGIBLE' for row in task['schedule'])
    return {'status': 'INTEGRITY_VALID', 'scheduled_windows': len(windows),
            'decision_rows': total, 'estimated_rows': estimated,
            'eligible_windows': eligible,
            'eligible_window_fraction': eligible / len(windows),
            'estimated_decision_fraction': estimated / total if total else None,
            'research_admissible': False}


def _control_burden(task, sources, scopes):
    windows = _validate_schedule(task['schedule'], task['method_ids'], sources, scopes,
                                 decisions=task['declarations'], decision_kind='declaration',
                                 expected_schedule=task['_expected_schedule'])
    intervals = task['exposure_intervals']
    if not isinstance(intervals, list):
        _fail('exposure interval ledger is required')
    by_unit = {}
    for row in intervals:
        _keys(row, ('exposure_id', 'physical_unit_id', 'source_ids', 'start_at', 'end_at'))
        ids = _ids(row['source_ids'])
        if not ids <= sources.keys() or not isinstance(row['exposure_id'], str) or not row['exposure_id'].strip():
            _fail('exposure interval has unknown membership')
        _same_physical_unit(ids, scopes, row['physical_unit_id'])
        start, end = _time(row['start_at']), _time(row['end_at'])
        if end <= start:
            _fail('exposure interval must have positive duration')
        by_unit.setdefault(row['physical_unit_id'], []).append((start, end))
    if len({r['exposure_id'] for r in intervals}) != len(intervals):
        _fail('duplicate exposure interval ID')
    total_seconds = 0.0
    for rows in by_unit.values():
        rows.sort()
        if any(right[0] < left[1] for left, right in zip(rows, rows[1:])):
            _fail('overlapping exposure intervals double-count physical follow-up')
        total_seconds += sum((end - start).total_seconds() for start, end in rows)
    declarations = {(r['window_id'], r['method_id']): r for r in task['declarations']}
    episode_ids = set()
    episode_counts = {method: 0 for method in task['method_ids']}
    linked_declarations = set()
    for episode in task['episodes']:
        _keys(episode, ('episode_id', 'physical_unit_id', 'method_id', 'declaration_window_id', 'source_ids', 'start_at', 'end_at'))
        if episode['episode_id'] in episode_ids:
            _fail('duplicate alert episode ID')
        episode_ids.add(episode['episode_id'])
        ids = _ids(episode['source_ids'])
        if not ids <= sources.keys() or episode['physical_unit_id'] not in by_unit:
            _fail('alert episode lacks source-backed exposure')
        _same_physical_unit(ids, scopes, episode['physical_unit_id'])
        key = (episode['declaration_window_id'], episode['method_id'])
        declaration = declarations.get(key)
        if declaration is None or declaration['status'] != 'ALARM_DECLARED' or key in linked_declarations:
            _fail('episode must link one actual, unique alarm declaration')
        linked_declarations.add(key)
        start, end = _time(episode['start_at']), _time(episode['end_at'])
        alarm_at = _time(declaration['declaration_at'])
        if not start <= alarm_at <= end:
            _fail('episode interval does not contain its declared alarm')
        if ids != set(declaration['source_ids']):
            _fail('episode sources do not match its actual declaration')
        intervals_for_unit = sorted(by_unit[episode['physical_unit_id']])
        if not any(left <= alarm_at < right for left, right in intervals_for_unit):
            _fail('alarm declaration is outside its own half-open exposure interval')
        covered_until = start
        for left, right in intervals_for_unit:
            if left <= covered_until < right:
                covered_until = right
            if covered_until >= end:
                break
        if covered_until < end:
            _fail('episode is outside its own continuous exposure support')
        episode_counts[episode['method_id']] += 1
    expected_alarms = {key for key, row in declarations.items() if row['status'] == 'ALARM_DECLARED'}
    if linked_declarations != expected_alarms:
        _fail('omitted episode for an actual alarm declaration')
    exposure_years = total_seconds / (86400.0 * _YEAR_DAYS)
    results = {}
    for method in task['method_ids']:
        if exposure_years <= 0:
            results[method] = {'status': 'NOT_ESTIMABLE', 'reason': 'NO_EXPOSURE_INTERVALS',
                               'value': None, 'episodes': episode_counts[method], 'exposure_years': None}
        else:
            results[method] = {'status': 'ESTIMATED', 'reason': None,
                               'value': episode_counts[method] / exposure_years,
                               'episodes': episode_counts[method], 'exposure_years': exposure_years}
    return {'status': 'INTEGRITY_VALID', 'scheduled_windows': len(windows),
            'exposure_intervals': len(intervals), 'by_method': results,
            'research_admissible': False}


def _event_cases(task, sources, scopes):
    windows = _validate_schedule(task['schedule'], task['method_ids'], sources, scopes,
                                 decisions=task['declarations'], decision_kind='declaration',
                                 expected_schedule=task['_expected_schedule'])
    horizon = task['warning_horizon_days']
    if type(horizon) is not int or horizon <= 0:
        _fail('positive explicit warning horizon required')
    events = {}
    for event in task['events']:
        _keys(event, ('event_id', 'physical_unit_id', 'source_ids', 'window_start', 'onset_earliest', 'onset_latest'))
        if not isinstance(event['event_id'], str) or not event['event_id'].strip() or event['event_id'] in events:
            _fail('duplicate or empty event ID')
        ids = _ids(event['source_ids'])
        if not ids <= sources.keys():
            _fail('event references unknown source')
        _same_physical_unit(ids, scopes, event['physical_unit_id'])
        start, earliest, latest = (_time(event[k], date_only=True) for k in ('window_start', 'onset_earliest', 'onset_latest'))
        if latest < earliest or start > earliest:
            _fail('event onset interval or observation window is reversed')
        events[event['event_id']] = (event, start, earliest, latest, ids)
    output = []
    for event_id, (event, start, earliest, latest, event_sources) in events.items():
        supporting_windows = []
        for wid, window in windows.items():
            ids = _ids(window['source_ids'], allow_empty=True)
            physical = {scopes[sid]['physical_unit_id'] for sid in ids if scopes[sid]['physical_status'] == 'KNOWN'}
            if event['physical_unit_id'] in physical and start <= _time(window['decision_at']) <= latest:
                supporting_windows.append((wid, window))
        if not supporting_windows:
            _fail('event case has no scheduled monitoring support')
        for method in task['method_ids']:
            rows = [(wid, task['declaration_index'][(wid, method)]) for wid, _ in supporting_windows]
            alarm_rows = sorted((row for _, row in rows if row['status'] == 'ALARM_DECLARED'),
                                key=lambda row: _time(row['declaration_at']))
            precursor_start = latest.timestamp() - horizon * 86400
            precursors = [r for r in alarm_rows if precursor_start <= _time(r['declaration_at']).timestamp() < earliest.timestamp()]
            if precursors:
                declaration = precursors[0]['declaration_at']
                decl_time = _time(declaration)
                output.append({'event_id': event_id, 'method_id': method, 'status': 'DETECTED',
                               'reason': None, 'declaration_at': declaration,
                               'lead_time_days': [(earliest - decl_time).total_seconds()/86400,
                                                  (latest - decl_time).total_seconds()/86400],
                               'source_ids': sorted(event_sources)})
                continue
            overlap = [r for r in alarm_rows if earliest.timestamp() <= _time(r['declaration_at']).timestamp() <= latest.timestamp()]
            unresolved = any(row['status'] in {'ABSTAINED', 'NOT_ESTIMABLE'} for _, row in rows)
            status = 'AMBIGUOUS' if overlap else ('NOT_ESTIMABLE' if unresolved else 'NOT_DETECTED')
            reason = 'ONSET_INTERVAL_OVERLAP' if overlap else ('INCOMPLETE_OBSERVATION' if unresolved else None)
            output.append({'event_id': event_id, 'method_id': method, 'status': status,
                           'reason': reason, 'declaration_at': None, 'lead_time_days': None,
                           'source_ids': sorted(event_sources)})
    return {'status': 'INTEGRITY_VALID', 'event_cases': output,
            'scheduled_windows': len(windows), 'research_admissible': False}


def _comparison(task, sources, scopes):
    if not isinstance(task['method_ids'], list) or len(task['method_ids']) != 2 or len(set(task['method_ids'])) != 2:
        _fail('comparison requires exactly two declared methods')
    results = []
    pair_ids = set()
    for pair in task['pairs']:
        _keys(pair, ('pair_id', 'source_ids', 'physical_unit_id', 'method_a', 'method_b', 'metric_id', 'status', 'value_a', 'value_b', 'reason'))
        if pair['pair_id'] in pair_ids:
            _fail('duplicate comparison pair ID')
        pair_ids.add(pair['pair_id'])
        if (pair['method_a'], pair['method_b']) != tuple(task['method_ids']) or not isinstance(pair['metric_id'], str) or not pair['metric_id'].strip():
            _fail('comparison method order or metric identity is invalid')
        ids = _ids(pair['source_ids'])
        if not ids <= sources.keys():
            _fail('comparison references unknown source')
        _same_physical_unit(ids, scopes, pair['physical_unit_id'])
        if pair['status'] == 'PAIRED':
            if not _finite_or_null(pair['value_a']) or not _finite_or_null(pair['value_b']) or pair['value_a'] is None or pair['value_b'] is None or pair['reason'] is not None:
                _fail('paired comparison requires two finite values and no missingness reason')
            delta = float(pair['value_b']) - float(pair['value_a'])
            result = {'status': 'PAIRED', 'reason': None, 'difference_b_minus_a': delta}
        elif pair['status'] == 'NOT_ESTIMABLE':
            if pair['value_a'] is not None or pair['value_b'] is not None or not isinstance(pair['reason'], str) or not pair['reason'].strip():
                _fail('unestimable comparison must preserve nulls and an explicit reason')
            result = {'status': 'NOT_ESTIMABLE', 'reason': pair['reason'], 'difference_b_minus_a': None}
        else:
            _fail('unknown comparison status')
        results.append({'pair_id': pair['pair_id'], 'physical_unit_id': pair['physical_unit_id'], **result})
    if not results:
        _fail('comparison task requires actual pair rows')
    return {'status': 'INTEGRITY_VALID', 'pairs': results,
            'known_physical_units': len({r['physical_unit_id'] for r in results if r['physical_unit_id'] is not None}),
            'independent_sampling_n': None, 'independence_status': 'NOT_ESTABLISHED',
            'unresolved_physical_pairs': sum(r['physical_unit_id'] is None for r in results),
            'research_admissible': False}


def _simulation(task, sources, scopes):
    if task['scope_role'] != 'development_validation' or not task['perturbations']:
        _fail('explicit simulations are limited to development validation fixtures')
    outputs = set()
    for row in task['perturbations']:
        _keys(row, ('perturbation_id', 'parent_source_ids', 'source_ids', 'transform', 'seed', 'parameters', 'input_sha256', 'output_sha256'))
        parents = _ids(row['parent_source_ids'])
        outputs_for_row = _ids(row['source_ids'])
        if not parents <= sources.keys() or not outputs_for_row <= sources.keys() or outputs_for_row & outputs:
            _fail('simulation source lineage is orphaned or duplicated')
        if any(sources[p]['origin'] == 'observational' for p in parents):
            _fail('simulation cannot consume observational research measurements')
        if row['transform'] not in {'mask_entries', 'add_declared_noise', 'scale_values', 'shift_time'}:
            _fail('unknown perturbation; no implicit simulation fallback')
        if type(row['seed']) is not int or row['seed'] < 0 or not isinstance(row['parameters'], dict) or not row['parameters']:
            _fail('simulation requires deterministic seed and explicit parameters')
        if not isinstance(row['input_sha256'], str) or not _SHA256.fullmatch(row['input_sha256']) or not isinstance(row['output_sha256'], str) or not _SHA256.fullmatch(row['output_sha256']):
            _fail('simulation input/output digest is invalid')
        parent_hashes = {sources[p]['sha256'] for p in parents}
        output_hashes = {sources[o]['sha256'] for o in outputs_for_row}
        if row['input_sha256'] not in parent_hashes or row['output_sha256'] not in output_hashes:
            _fail('simulation digest does not join to its actual source rows')
        if any(sources[o]['origin'] != 'simulation' for o in outputs_for_row):
            _fail('perturbed output source must retain simulation origin')
        outputs.update(outputs_for_row)
    return {'status': 'INTEGRITY_VALID', 'perturbations': len(task['perturbations']),
            'simulation_output_sources': sorted(outputs), 'research_admissible': False}


def validate_task(task, source_rows, scope_membership, *, expected_schedule=None):
    """Validate one versioned task and independently recompute its supported rows."""
    digest(task)
    digest(source_rows)
    digest(scope_membership)
    if not isinstance(task, dict) or task.get('task_type') not in {
            'OBSERVABILITY', 'CONTROL_BURDEN', 'EVENT_CASES', 'COMPARISON', 'SIMULATION'}:
        _fail('unknown task type')
    name = task['task_type'].lower()
    try:
        schema = json.loads((SCHEMA_DIR / (name + '.json')).read_text())
    except OSError as error:
        raise AdmissionError('task schema is unavailable') from error
    try:
        schema_check(task, schema)
    except AdmissionError as error:
        for field, rule in schema.get('properties', {}).items():
            if field in task:
                try:
                    schema_check(task[field], rule)
                except AdmissionError as field_error:
                    raise AdmissionError(f'task.{field}: {field_error}') from field_error
        raise AdmissionError(f'task: {error}') from error
    if task['schema_version'] != 1 or task['origin'] not in ORIGINS or task['scope_role'] not in SCOPE_ROLES:
        _fail('unsupported task schema, origin or scope role')
    sources, scopes = _validate_inputs(task, source_rows, scope_membership)
    if task['task_type'] in {'OBSERVABILITY', 'CONTROL_BURDEN', 'EVENT_CASES'}:
        task = dict(task)
        task['_expected_schedule'] = expected_schedule
    if task['task_type'] == 'OBSERVABILITY':
        return _observability(task, sources, scopes)
    if task['task_type'] == 'CONTROL_BURDEN':
        return _control_burden(task, sources, scopes)
    if task['task_type'] == 'EVENT_CASES':
        task = dict(task)
        task['declaration_index'] = {(r['window_id'], r['method_id']): r for r in task['declarations']}
        return _event_cases(task, sources, scopes)
    if task['task_type'] == 'COMPARISON':
        return _comparison(task, sources, scopes)
    return _simulation(task, sources, scopes)
