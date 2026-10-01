"""Retrospective as-of context selection, never historical execution evidence."""
from dataclasses import dataclass
from datetime import date, timedelta
from .features import Window


@dataclass(frozen=True)
class AsOfSelection:
    as_of_date: str
    status: str
    window: Window | None
    mode: str = "RETROSPECTIVE_AS_OF_SIMULATION"


def schedule_as_of(windows, as_of_dates):
    """Select the freshest newly available closed context at each scheduled date.

    Policy v1: one context per date, no reuse, no backfill of older contexts
    after a newer context was selected. Late batches cannot create multiple
    sustaining crossings at the same date. No scores or labels affect selection.
    Empty selections retain abstention rows. These dates are simulated as-of
    dates; a production alarm still needs actual execution/declaration evidence,
    source verification, freshness/QA eligibility and an episode/exposure ledger.
    """
    if isinstance(windows, (str, bytes)) or isinstance(as_of_dates, (str, bytes)):
        raise ValueError("provide collections of windows and scheduled dates")
    windows = tuple(windows)
    dates = tuple(date.fromisoformat(x) for x in as_of_dates)
    if not windows or not dates:
        raise ValueError("nonempty closed windows and schedule required")
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError("scheduled dates must be unique and strictly increasing")
    prior_end = -1
    anchor = None
    availability = []
    for window in windows:
        if not isinstance(window, Window):
            raise ValueError("schedule accepts explicit Window descriptors")
        start, end, earliest = (date.fromisoformat(x) for x in
            (window.start_date, window.latest_observation_date, window.decision_date))
        if (type(window.start_index) is not int or type(window.stop_index) is not int
                or window.start_index < 0 or window.stop_index - window.start_index < 2
                or (end-start).days+1 != window.stop_index-window.start_index
                or earliest < end or window.stop_index <= prior_end):
            raise ValueError("windows must have consistent calendar support and increasing context ends")
        current_anchor = start - timedelta(days=window.start_index)
        if anchor is not None and current_anchor != anchor:
            raise ValueError("windows must refer to the same indexed daily calendar")
        anchor = current_anchor
        prior_end = window.stop_index
        availability.append(earliest)
    selected_end = -1
    selections = []
    for as_of in dates:
        candidates = [window for window, earliest in zip(windows, availability)
                      if earliest <= as_of and window.stop_index > selected_end]
        chosen = candidates[-1] if candidates else None
        if chosen:
            selected_end = chosen.stop_index
        selections.append(AsOfSelection(as_of.isoformat(),
            "SELECTED_CLOSED_CONTEXT" if chosen else "NO_NEW_AVAILABLE_CONTEXT", chosen))
    return tuple(selections)
