"""Shared number ledger for the manuscript builder and its verifier.

Every number printed in the manuscript is produced by ``Ledger.num`` (a value at
a packet path), ``Ledger.const`` (a protocol constant read from code) or
``Ledger.claim`` (a qualitative statement with a boolean expression over packet
paths). The ledger is saved next to the manuscript; the verifier re-resolves
every path against a freshly recomputed packet and re-imports every constant.
"""
from __future__ import annotations

import importlib
import inspect
import math
import re


def resolve(packet, path: str):
    """Resolve 'a.b[k=v,k2=v2].c[0]' against nested dicts/lists. Keys may contain '|'."""
    value = packet
    for part in re.findall(r"\['[^']*'\]|[^.\[\]]+|\[[^\]]*\]", path):
        if part.startswith("['"):
            value = value[part[2:-2]]
            continue
        if part.startswith("["):
            sel = part[1:-1]
            if re.fullmatch(r"-?\d+", sel):
                value = value[int(sel)]
            else:
                conds = dict(c.split("=", 1) for c in sel.split(","))
                hits = [v for v in value if all(str(v.get(k)) == x for k, x in conds.items())]
                if len(hits) != 1:
                    raise KeyError(f"selector {sel} matched {len(hits)} items in {path}")
                value = hits[0]
        else:
            value = value[part]
    return value


def fmt(value, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError("cannot print a missing value as a number")
    if spec == "int":
        return f"{int(round(value)):,}"
    if spec.startswith("pct"):
        return f"{100 * value:.{int(spec[3:] or 0)}f}%"
    if spec.startswith("signed"):
        digits = int(spec[6:] or 3)
        out = f"{value:+.{digits}f}"
        return out.replace("-", "−")
    out = format(value, spec)
    return out.replace("-", "−") if out.startswith("-") else out


def const_value(module: str, attr: str):
    if ":" in attr:  # function default, e.g. "train_factorized_sr1_model:lr"
        func, arg = attr.split(":")
        return inspect.signature(getattr(importlib.import_module(module), func)).parameters[arg].default
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


class Ledger:
    def __init__(self, packet):
        self.packet = packet
        self.entries: list[dict] = []

    def num(self, path: str, spec: str = ".3f", scale: float = 1.0) -> str:
        value = resolve(self.packet, path)
        text = fmt(value * scale, spec)
        self.entries.append({"kind": "packet", "path": path, "spec": spec, "scale": scale, "text": text})
        return text

    def const(self, module: str, attr: str, spec: str = "g", scale: float = 1.0) -> str:
        value = const_value(module, attr)
        text = fmt(value * scale, spec)
        self.entries.append({"kind": "const", "module": module, "attr": attr, "spec": spec, "scale": scale, "text": text})
        return text

    def claim(self, expression: str, description: str) -> bool:
        result = bool(evaluate(self.packet, expression))
        self.entries.append({"kind": "claim", "expression": expression, "description": description, "value": result})
        return result


def evaluate(packet, expression: str):
    env = {"R": lambda p: resolve(packet, p), "abs": abs, "min": min, "max": max, "len": len, "all": all, "any": any,
           "sum": sum}
    return eval(expression, {"__builtins__": {}, **env})  # expressions are written in this repository only
