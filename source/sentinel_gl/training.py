"""Small masked-learning engine. Successful optimization is not scientific evidence."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import inspect
import random
import time
from pathlib import Path
import numpy as np
import torch
from .model import TimeSeriesMAE, evaluation_masks


@dataclass(frozen=True)
class StopRule:
    min_epochs: int
    max_epochs: int
    patience: int
    min_delta: float

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.min_epochs,self.max_epochs,self.patience)):
            raise ValueError("epoch bounds and patience must be positive integers")
        if self.max_epochs < self.min_epochs or isinstance(self.min_delta, bool) or not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("invalid stop rule")


def seed_everything(seed: int):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validation_loss(model, batches, device):
    """Observed-time balanced cross masks do not draw new validation randomness."""
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for x, valid in batches:
            x, valid = x.to(device), valid.to(device)
            model._validate_input(x,valid)
            for mask in evaluation_masks(valid, 2):
                pred,_ = model.reconstruct(x,mask,valid)
                targets = mask.unsqueeze(-1) & valid
                total += float(((pred-x).square()*targets).sum().item())
                count += int(targets.sum().item())
    if not count or not np.isfinite(total):
        raise ValueError("validation loss is not estimable")
    return total/count


def fit_masked_autoencoder(*, model_config, train_batches, validation_batches,
                          seed: int, output_dir, stop_rule: StopRule,
                          transform_state, device: str, learning_rate: float,
                          weight_decay: float, max_grad_norm: float):
    """Build after seeding, train on supplied tensors, preserve exact architecture.

    Data lineage, temporal cutoffs and split membership must be verified by the
    caller/domain adapter. No acquisition, cohort selection or publication
    claim is performed here. The output directory must not already exist.
    Validation batches must be re-iterable; generators are rejected.
    """
    if device not in {"cpu","mps","cuda"}:
        raise ValueError("device must be explicitly cpu, mps or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    if not isinstance(train_batches,(list,tuple)) or not isinstance(validation_batches,(list,tuple)) or not train_batches or not validation_batches:
        raise ValueError("nonempty re-iterable train and validation batch collections are required")
    required = {"mean","scale","fit_lake_ids","constant_channels"}
    if set(transform_state) != required or not transform_state["fit_lake_ids"]:
        raise ValueError("explicit fitted normalization state is required")
    if any(isinstance(x,bool) or not np.isfinite(x) for x in (learning_rate,weight_decay,max_grad_norm)) or learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("invalid optimization settings")
    seed_everything(seed)
    configuration = inspect.signature(TimeSeriesMAE).bind(**model_config)
    configuration.apply_defaults()
    model_config = dict(configuration.arguments)
    model = TimeSeriesMAE(**model_config).to(device)
    mean = np.asarray(transform_state["mean"])
    scale = np.asarray(transform_state["scale"])
    if mean.dtype.kind not in "iuf" or scale.dtype.kind not in "iuf":
        raise ValueError("normalization metadata must contain real numeric values, not strings or booleans")
    mean, scale = mean.astype(float), scale.astype(float)
    if mean.shape != (model.n_channels,) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("normalization metadata must match channels and have finite positive scales")
    for x,valid in (*train_batches,*validation_batches):
        model._validate_input(x.to(device),valid.to(device))
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    best, significant_best, best_epoch, bad = float("inf"),float("inf"),0,0
    history = []
    start = time.perf_counter()
    for epoch in range(1,stop_rule.max_epochs+1):
        model.train()
        total,count = 0.,0
        for x,valid in train_batches:
            x,valid=x.to(device),valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            result=model(x,validity=valid)
            result["loss"].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm,error_if_nonfinite=True)
            optimizer.step()
            n=result["target_count"]
            total+=float(result["loss"].item())*n;count+=n
        if not count or not np.isfinite(total):
            raise ValueError("training loss is not estimable")
        val=validation_loss(model,validation_batches,device)
        row={"epoch":epoch,"train_loss":total/count,"validation_loss":val,"target_count":count}
        history.append(row)
        with (out/'history.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        if val < best:
            best,best_epoch=val,epoch
            checkpoint={"format_version":2,"model_config":dict(model_config),
                        "model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                        "normalization":dict(transform_state),"seed":seed,"epoch":epoch,
                        "selection":"minimum_observed_validation_loss","validation_loss":val,
                        "training_mask_policy":"observed_target_and_visible_context_v1",
                        "evaluation_mask_policy":"observed_time_balanced_cross_masks_v1",
                        "stop_rule":asdict(stop_rule),"torch_rng_state":torch.get_rng_state()}
            torch.save(checkpoint,out/'checkpoint_best.pt')
        if val < significant_best-stop_rule.min_delta:
            significant_best,bad=val,0
        else:
            bad+=1
        if epoch >= stop_rule.min_epochs and bad >= stop_rule.patience:
            status="EARLY_STOPPED"
            break
    else:
        status="BUDGET_EXHAUSTED"
    checkpoint=torch.load(out/'checkpoint_best.pt',weights_only=True,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    summary={"status":status,"seed":seed,"model_config":dict(model_config),
             "stop_rule":asdict(stop_rule),"epochs_trained":len(history),
             "checkpoint_epoch":best_epoch,"best_validation_loss":best,
             "elapsed_wall_seconds":time.perf_counter()-start,"device":device,
             "batch_sizes":[int(x.shape[0]) for x,_ in train_batches],
             "learning_rate":learning_rate,"weight_decay":weight_decay,
             "scientific_claims":[],"history":history}
    summary["split_integrity"] = "CALLER_LINEAGE_VERIFICATION_REQUIRED"
    summary["resume_supported"] = False
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return model,summary
