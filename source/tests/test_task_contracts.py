"""Task contracts are exercised only with declared analytical rows."""
from copy import deepcopy
import hashlib

import pytest

from sentinel_gl.admission import AdmissionError
from sentinel_gl.task_contracts import validate_sampling_provenance_graph, validate_task


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def inputs(ids=('s1',), *, role='evaluation', units=None, statuses=None):
    units = units or {item: ('lake-a' if item != 'unknown' else None) for item in ids}
    statuses = statuses or {}
    sources, scopes = {}, {}
    for item in ids:
        status = statuses.get(item, 'MEASURED')
        source = {'source_id': item, 'sha256': digest(item), 'origin': 'fixture', 'status': status}
        if status == 'MEASURED':
            source.update(support_start='2020-01-01T00:00:00Z',
                          support_end='2021-01-01T00:00:00Z',
                          available_at='2021-01-02T00:00:00Z')
        else:
            source['reason'] = 'ANALYTICAL_FIXTURE_MISSING'
        sources[item] = source
        scopes[item] = {'source_id': item, 'scope_role': role,
                        'physical_status': 'UNKNOWN' if units[item] is None else 'KNOWN',
                        'physical_unit_id': units[item]}
    return sources, scopes


def window(wid='w1', *, source_ids=('s1',), eligibility='ELIGIBLE', reason=None,
           decision='2021-01-05T00:00:00Z'):
    return {'window_id': wid, 'source_ids': list(source_ids),
            'support_start': '2020-01-01T00:00:00Z',
            'support_end': '2021-01-01T00:00:00Z', 'decision_at': decision,
            'eligibility_status': eligibility, 'reason': reason}


def task_base(task_type, source_ids, **values):
    return {'schema_version': 1, 'task_id': 'analytic-task', 'task_type': task_type,
            'origin': 'fixture', 'scope_role': 'evaluation',
            'source_ids': list(source_ids), **values}


def test_observability_recomputes_poor_outcome_and_preserves_abstention():
    sources, scopes = inputs(('s1',))
    schedule = [window()]
    task = task_base('OBSERVABILITY', ['s1'], method_ids=['method'], schedule=schedule,
                     decisions=[{'window_id': 'w1', 'method_id': 'method', 'source_ids': ['s1'],
                                 'status': 'ABSTAINED', 'value': None, 'reason': 'INSUFFICIENT_INPUT'}])
    result = validate_task(task, sources, scopes, expected_schedule=deepcopy(schedule))
    assert result['status'] == 'INTEGRITY_VALID'
    assert result['estimated_decision_fraction'] == 0.0
    assert result['eligible_window_fraction'] == 1.0
    assert result['research_admissible'] is False


def test_observability_rejects_missing_schedule_decision_null_reason_and_wrong_scope():
    sources, scopes = inputs(('s1',))
    schedule = [window()]
    valid = task_base('OBSERVABILITY', ['s1'], method_ids=['method'], schedule=schedule,
                      decisions=[{'window_id': 'w1', 'method_id': 'method', 'source_ids': ['s1'],
                                  'status': 'NOT_ESTIMABLE', 'value': None, 'reason': 'MASKED'}])
    with pytest.raises(AdmissionError, match='schedule'):
        validate_task(valid, sources, scopes)
    with pytest.raises(AdmissionError, match='schedule'):
        validate_task({**valid, 'schedule': []}, sources, scopes, expected_schedule=schedule)
    for decisions in ([], [{**valid['decisions'][0], 'reason': None}],
                      [{**valid['decisions'][0], 'status': 'ESTIMATED', 'value': None, 'reason': None}],
                      [{**valid['decisions'][0], 'source_ids': []}]):
        with pytest.raises(AdmissionError):
            validate_task({**valid, 'decisions': decisions}, sources, scopes, expected_schedule=schedule)
    wrong_scope = deepcopy(scopes)
    wrong_scope['s1']['scope_role'] = 'fitting'
    with pytest.raises(AdmissionError, match='scope'):
        validate_task(valid, sources, wrong_scope, expected_schedule=schedule)


def burden_task(episode=True):
    sources, scopes = inputs(('s1',), role='evaluation')
    schedule = [window(decision='2021-01-05T00:00:00Z')]
    declarations = [{'window_id': 'w1', 'method_id': 'method', 'source_ids': ['s1'],
                     'status': 'ALARM_DECLARED' if episode else 'NO_ALARM',
                     'declaration_at': '2021-01-05T00:00:00Z' if episode else None, 'reason': None}]
    episodes = ([{'episode_id': 'ep1', 'physical_unit_id': 'lake-a', 'method_id': 'method',
                  'declaration_window_id': 'w1', 'source_ids': ['s1'],
                  'start_at': '2021-01-03T00:00:00Z', 'end_at': '2021-01-06T00:00:00Z'}]
                if episode else [])
    task = task_base('CONTROL_BURDEN', ['s1'], method_ids=['method'], schedule=schedule,
                     declarations=declarations,
                     exposure_intervals=[{'exposure_id': 'x1', 'physical_unit_id': 'lake-a',
                                          'source_ids': ['s1'], 'start_at': '2021-01-03T00:00:00Z',
                                          'end_at': '2021-01-06T00:00:00Z'}], episodes=episodes)
    return task, sources, scopes, schedule


def test_control_burden_recomputes_from_episode_and_exposure_rows():
    task, sources, scopes, schedule = burden_task()
    result = validate_task(task, sources, scopes, expected_schedule=schedule)
    assert result['by_method']['method']['status'] == 'ESTIMATED'
    assert result['by_method']['method']['episodes'] == 1
    assert result['by_method']['method']['value'] == pytest.approx(365.2425 / 3)
    poor, sources, scopes, schedule = burden_task(episode=False)
    assert validate_task(poor, sources, scopes, expected_schedule=schedule)['by_method']['method']['value'] == 0.0
    with pytest.raises(AdmissionError, match='declaration'):
        validate_task({**task, 'episodes': [{**task['episodes'][0], 'declaration_window_id': 'orphan'}]},
                      sources, scopes, expected_schedule=schedule)


def event_task(declaration='ALARM_DECLARED'):
    sources, scopes = inputs(('s1', 'event'), units={'s1': 'lake-a', 'event': 'lake-a'})
    schedule = [window(decision='2021-01-04T00:00:00Z')]
    declared = declaration == 'ALARM_DECLARED'
    task = task_base('EVENT_CASES', ['s1', 'event'], method_ids=['method'], schedule=schedule,
                     declarations=[{'window_id': 'w1', 'method_id': 'method', 'source_ids': ['s1'],
                                    'status': declaration, 'declaration_at': '2021-01-04T00:00:00Z' if declared else None,
                                    'reason': None if declaration == 'NO_ALARM' else ('MISSING' if not declared else None)}],
                     events=[{'event_id': 'event-1', 'physical_unit_id': 'lake-a', 'source_ids': ['event'],
                              'window_start': '2021-01-01', 'onset_earliest': '2021-01-05',
                              'onset_latest': '2021-01-07'}], warning_horizon_days=10)
    return task, sources, scopes, schedule


def test_event_cases_recompute_uncertain_lead_and_accept_valid_miss():
    task, sources, scopes, schedule = event_task()
    detected = validate_task(task, sources, scopes, expected_schedule=schedule)['event_cases'][0]
    assert detected['status'] == 'DETECTED'
    assert detected['lead_time_days'] == [1, 3]
    miss, sources, scopes, schedule = event_task('NO_ALARM')
    result = validate_task(miss, sources, scopes, expected_schedule=schedule)['event_cases'][0]
    assert result['status'] == 'NOT_DETECTED'
    assert result['lead_time_days'] is None and result['declaration_at'] is None


def test_comparison_is_paired_arithmetic_not_a_superiority_claim():
    sources, scopes = inputs(('s1',))
    task = task_base('COMPARISON', ['s1'], method_ids=['a', 'b'], pairs=[{
        'pair_id': 'p1', 'source_ids': ['s1'], 'physical_unit_id': 'lake-a',
        'method_a': 'a', 'method_b': 'b', 'metric_id': 'declared-metric',
        'status': 'PAIRED', 'value_a': 5.0, 'value_b': 2.0, 'reason': None}])
    result = validate_task(task, sources, scopes)
    assert result['pairs'][0]['difference_b_minus_a'] == -3.0
    assert 'p_value' not in result and 'superior' not in result
    corrupted = deepcopy(task)
    corrupted['pairs'][0].update(status='NOT_ESTIMABLE', value_a=0.0, value_b=None, reason='MISSING')
    with pytest.raises(AdmissionError, match='nulls'):
        validate_task(corrupted, sources, scopes)


def test_simulation_requires_explicit_origin_parent_hash_and_deterministic_seed():
    sources, scopes = inputs(('base', 'sim'), role='development_validation',
                             units={'base': None, 'sim': None})
    sources['sim'].update(origin='simulation', status='DERIVED', reason='declared analytical transform')
    task = {'schema_version': 1, 'task_id': 'sim-task', 'task_type': 'SIMULATION',
            'origin': 'simulation', 'scope_role': 'development_validation',
            'source_ids': ['base', 'sim'], 'method_ids': ['simulator'],
            'perturbations': [{'perturbation_id': 'p1', 'parent_source_ids': ['base'],
                               'source_ids': ['sim'], 'transform': 'mask_entries', 'seed': 7,
                               'parameters': {'fraction': 0.25}, 'input_sha256': digest('base'),
                               'output_sha256': digest('sim')}]}
    result = validate_task(task, sources, scopes)
    assert result['perturbations'] == 1 and result['research_admissible'] is False
    for changed in ({**task, 'origin': 'fixture'},
                    {**task, 'perturbations': [{**task['perturbations'][0], 'seed': None}]},
                    {**task, 'perturbations': [{**task['perturbations'][0], 'output_sha256': '0' * 64}]}):
        with pytest.raises(AdmissionError):
            validate_task(changed, sources, scopes)


def test_sampling_and_provenance_graphs_do_not_conflate_checkpoints_seeds_or_unknowns():
    ids = ('s1', 's2', 's3', 'unknown')
    units = {'s1': 'lake-a', 's2': 'lake-a', 's3': 'lake-b', 'unknown': None}
    sources, scopes = inputs(ids, role='evaluation', units=units)
    graph = {'G_S': {'nodes': [
        {'node_id': 'n1', 'source_ids': ['s1'], 'physical_status': 'KNOWN', 'physical_unit_id': 'lake-a'},
        {'node_id': 'n2', 'source_ids': ['s2'], 'physical_status': 'KNOWN', 'physical_unit_id': 'lake-a'},
        {'node_id': 'n3', 'source_ids': ['s3'], 'physical_status': 'KNOWN', 'physical_unit_id': 'lake-b'},
        {'node_id': 'n4', 'source_ids': ['unknown'], 'physical_status': 'UNKNOWN', 'physical_unit_id': None}],
        'edges': [{'source': 'n1', 'target': 'n2', 'relation': 'same_physical_unit',
                   'evidence_source_ids': ['s1', 's2']}]},
        'G_P': {'nodes': [{'node_id': 'run-a', 'kind': 'run', 'source_ids': []},
                          {'node_id': 'run-b', 'kind': 'run', 'source_ids': []},
                          {'node_id': 'checkpoint', 'kind': 'checkpoint', 'source_ids': []},
                          {'node_id': 'seed-a', 'kind': 'seed', 'source_ids': []},
                          {'node_id': 'seed-b', 'kind': 'seed', 'source_ids': []}],
                'edges': [{'source': 'run-a', 'target': 'checkpoint', 'relation': 'checkpoint_of'},
                          {'source': 'run-b', 'target': 'checkpoint', 'relation': 'checkpoint_of'},
                          {'source': 'seed-a', 'target': 'run-a', 'relation': 'seed_for'},
                          {'source': 'seed-b', 'target': 'run-b', 'relation': 'seed_for'}]}}
    result = validate_sampling_provenance_graph(graph, sources, scopes)
    assert result['known_physical_units'] == 2
    assert result['declared_sampling_n'] == 2
    assert result['independent_sampling_n'] is None and result['independence_status'] == 'NOT_ESTABLISHED'
    assert result['seed_count'] == 2
    assert result['unresolved_source_ids'] == ['unknown']
    with pytest.raises(AdmissionError, match='provenance relation'):
        bad = deepcopy(graph)
        bad['G_S']['edges'].append({'source': 'n1', 'target': 'n2', 'relation': 'seed_for', 'evidence_source_ids': ['s1']})
        validate_sampling_provenance_graph(bad, sources, scopes)

    with pytest.raises(AdmissionError, match='unknown physical'):
        bad = deepcopy(graph)
        bad['G_S']['edges'].append({'source': 'n1', 'target': 'n4', 'relation': 'shared_waterbody', 'evidence_source_ids': ['s1']})
        validate_sampling_provenance_graph(bad, sources, scopes)
    bad = deepcopy(graph)
    bad['G_S']['nodes'] = bad['G_S']['nodes'][:-1]
    with pytest.raises(AdmissionError, match='every independently'):
        validate_sampling_provenance_graph(bad, sources, scopes)


def test_episode_cannot_be_backdated_omitted_or_outside_exposure():
    task, sources, scopes, schedule = burden_task()
    for changes, message in (({'declarations': [{**task['declarations'][0], 'declaration_at':'2021-01-04T00:00:00Z'}]}, 'arrival'),
                             ({'episodes': []}, 'omitted episode'),
                             ({'exposure_intervals': [{**task['exposure_intervals'][0], 'start_at':'2020-01-01T00:00:00Z','end_at':'2021-01-01T00:00:00Z'}]}, 'alarm declaration is outside its own half-open exposure interval')):
        with pytest.raises(AdmissionError, match=message):
            validate_task({**task, **changes}, sources, scopes, expected_schedule=schedule)


def test_orphan_alarm_is_a_contract_error_and_availability_boundary_is_inclusive():
    task, sources, scopes, schedule = event_task()
    sources['s1']['available_at'] = schedule[0]['decision_at']
    assert validate_task(task,sources,scopes,expected_schedule=schedule)['event_cases'][0]['status']=='DETECTED'
    bad=deepcopy(task);bad['declarations'][0]['window_id']='orphan'
    with pytest.raises(AdmissionError,match='orphan'):
        validate_task(bad,sources,scopes,expected_schedule=schedule)


def test_event_lead_uses_actual_time_and_onset_overlap_is_ambiguous():
    task,sources,scopes,schedule=event_task()
    schedule[0]['decision_at']='2021-01-04T12:00:00Z'
    task['declarations'][0]['declaration_at']=schedule[0]['decision_at']
    result=validate_task(task,sources,scopes,expected_schedule=schedule)['event_cases'][0]
    assert result['lead_time_days']==[.5,2.5]
    schedule[0]['decision_at']='2021-01-06T00:00:00Z'
    task['declarations'][0]['declaration_at']=schedule[0]['decision_at']
    result=validate_task(task,sources,scopes,expected_schedule=schedule)['event_cases'][0]
    assert result['status']=='AMBIGUOUS' and result['lead_time_days'] is None


def test_task_rejects_nonfinite_nested_simulation_parameters():
    task,sources,scopes,schedule=event_task()
    task['unexpected']={'nonfinite':float('nan')}
    with pytest.raises(ValueError):
        validate_task(task,sources,scopes,expected_schedule=schedule)


def test_zero_duration_episode_still_requires_exposed_declaration_time():
    task,sources,scopes,schedule=burden_task()
    task['episodes'][0].update(start_at='2021-01-05T00:00:00Z',end_at='2021-01-05T00:00:00Z')
    assert validate_task(task,sources,scopes,expected_schedule=schedule)['by_method']['method']['episodes']==1
    task['exposure_intervals'][0]['end_at']='2021-01-05T00:00:00Z'
    with pytest.raises(AdmissionError,match='half-open exposure'):
        validate_task(task,sources,scopes,expected_schedule=schedule)
