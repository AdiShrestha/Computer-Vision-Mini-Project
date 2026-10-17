"""Constructed checkpoint eligibility regression; no empirical training."""
from sentinel_gl.checkpoint_selection import CheckpointSelector

def test_checkpoint_selector_honors_minimum_eligibility_and_null_monitor_semantics():
    selector = CheckpointSelector(
        minimum_optimization_epochs=4,
        earliest_checkpoint_epoch=4,
        patience=2,
        tolerance=0.1,
    )
    early = selector.observe(1, 0.01, 50)
    assert not early["eligible"] and selector.best_epoch is None
    assert selector.nonimprovement_count == 0
    first = selector.observe(4, 2.0, 50)
    assert first["selected"] and selector.best_epoch == 4
    missing = selector.observe(5, None, 0)
    assert not missing["eligible"] and selector.nonimprovement_count == 0
    tie = selector.observe(6, 2.05, 50)
    assert not tie["selected"] and tie["reason"] == "TIE_WITHIN_TOLERANCE_RETAIN_EARLIER"
    assert selector.best_epoch == 4 and selector.nonimprovement_count == 1
    stop = selector.observe(7, 2.2, 50)
    assert stop["stop"] and selector.best_epoch == 4

    unavailable = CheckpointSelector(
        minimum_optimization_epochs=2,
        earliest_checkpoint_epoch=2,
        patience=1,
        tolerance=0.0,
    )
    for epoch in range(1, 4):
        unavailable.observe(epoch, None, 0)
    assert unavailable.best_epoch is None
    assert unavailable.selected_status == "UNAVAILABLE_NO_ELIGIBLE_FINITE_MONITOR_ESTIMATE"
