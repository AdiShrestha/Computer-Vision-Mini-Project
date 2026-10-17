"""Finite-support checkpoint selection with explicit epoch eligibility."""
from __future__ import annotations
import math

class CheckpointSelector:
    """Finite, denominator-supported monitor selection with explicit epoch rules."""
    def __init__(self, *, minimum_optimization_epochs: int,
                 earliest_checkpoint_epoch: int, patience: int, tolerance: float):
        if (type(minimum_optimization_epochs) is not int or minimum_optimization_epochs < 1 or
            type(earliest_checkpoint_epoch) is not int or earliest_checkpoint_epoch < minimum_optimization_epochs or
            type(patience) is not int or patience < 1 or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("checkpoint rule requires positive budgets/patience, earliest>=minimum and finite nonnegative tolerance")
        self.minimum_optimization_epochs=minimum_optimization_epochs
        self.earliest_checkpoint_epoch=earliest_checkpoint_epoch
        self.patience=patience; self.tolerance=float(tolerance)
        self.best_loss: float | None=None; self.best_epoch: int | None=None
        self.nonimprovement_count=0; self.eligible_count=0

    def observe(self, epoch: int, loss: float | None, denominator: int) -> dict:
        if type(epoch) is not int or epoch < 1 or type(denominator) is not int or denominator < 0:
            raise ValueError("checkpoint observation epoch/denominator is invalid")
        if epoch < self.earliest_checkpoint_epoch:
            return {"eligible":False,"selected":False,"reason":"BEFORE_EARLIEST_CHECKPOINT_EPOCH",
                    "nonimprovement_count":self.nonimprovement_count}
        if denominator == 0 or loss is None or not math.isfinite(float(loss)):
            return {"eligible":False,"selected":False,
                    "reason":"NO_FINITE_POSITIVE_DENOMINATOR_MONITOR_ESTIMATE",
                    "nonimprovement_count":self.nonimprovement_count}
        self.eligible_count+=1
        value=float(loss)
        if self.best_loss is None or value < self.best_loss-self.tolerance:
            self.best_loss=value; self.best_epoch=epoch; self.nonimprovement_count=0
            reason="FIRST_ELIGIBLE_FINITE_ESTIMATE" if self.eligible_count==1 else "IMPROVEMENT_BEYOND_TOLERANCE"
            selected=True
        else:
            # Exact/within-tolerance ties retain the earlier checkpoint and count
            # as a non-improvement. Missing/nonfinite estimates never consume patience.
            self.nonimprovement_count+=1; selected=False
            reason="TIE_WITHIN_TOLERANCE_RETAIN_EARLIER" if abs(value-self.best_loss)<=self.tolerance else "NO_IMPROVEMENT"
        stop=(epoch>=self.minimum_optimization_epochs and
              self.nonimprovement_count>=self.patience)
        return {"eligible":True,"selected":selected,"reason":reason,
            "nonimprovement_count":self.nonimprovement_count,"stop":stop}

    @property
    def selected_status(self) -> str:
        return "SELECTED_ELIGIBLE_FINITE_MONITOR_CHECKPOINT" if self.best_epoch is not None else "UNAVAILABLE_NO_ELIGIBLE_FINITE_MONITOR_ESTIMATE"
