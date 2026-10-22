"""The tuning search space: a YAML file of knobs the pipeline accepts, each with a finite list of values.

    knobs:
      chunk_size: [20, 40, 80]
      top_k: [1, 3, 5]
      hybrid_weight: {min: 0.0, max: 1.0, step: 0.25}   # expands to 0.0, 0.25, ..., 1.0
      reranker: [none, overlap]
    baseline:            # optional; defaults to the first value of each knob
      chunk_size: 40
      top_k: 1

Every config a strategy proposes must assign every knob a value from its
list; `contains` checks that.
"""

import itertools
import json
import random
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

from ragaudit import yaml_io

MAX_RANGE_VALUES = 1000


class SearchSpaceError(ValueError):
    """Raised for a malformed search-space file or an out-of-space config."""


def _expand_range(knob: str, spec: dict) -> list:
    try:
        low, high, step = spec["min"], spec["max"], spec["step"]
    except KeyError as exc:
        raise SearchSpaceError(f"knob {knob!r}: a range needs min, max and step") from exc
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high, step)):
        raise SearchSpaceError(f"knob {knob!r}: min, max and step must be numbers")
    if step <= 0 or high < low:
        raise SearchSpaceError(f"knob {knob!r}: need step > 0 and max >= min")
    count = int(round((high - low) / step)) + 1
    if count > MAX_RANGE_VALUES:
        raise SearchSpaceError(f"knob {knob!r}: range expands to {count} values (max {MAX_RANGE_VALUES})")
    integral = all(isinstance(v, int) for v in (low, high, step))
    return [low + i * step if integral else round(low + i * step, 10) for i in range(count)]


def _same_value(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) < 1e-9
    return a == b


def canonical(config: dict) -> str:
    """A stable string identity for a config, for "already tried?" checks."""
    return json.dumps(config, sort_keys=True, default=str)


class SearchSpace(BaseModel):
    knobs: dict[str, list]
    baseline: dict

    @property
    def size(self) -> int:
        size = 1
        for values in self.knobs.values():
            size *= len(values)
        return size

    def normalize(self, config: dict) -> dict:
        """Return config with each value replaced by the space's own value; raise if anything is outside the space."""
        if set(config) != set(self.knobs):
            missing, extra = set(self.knobs) - set(config), set(config) - set(self.knobs)
            raise SearchSpaceError(f"config must set exactly the declared knobs (missing {sorted(missing)}, extra {sorted(extra)})")
        normalized = {}
        for knob, values in self.knobs.items():
            match = next((v for v in values if _same_value(v, config[knob])), None)
            if match is None and not any(_same_value(v, config[knob]) for v in values):
                raise SearchSpaceError(f"{knob}={config[knob]!r} is outside the search space {values}")
            normalized[knob] = match
        return normalized

    def contains(self, config: dict) -> bool:
        try:
            self.normalize(config)
        except SearchSpaceError:
            return False
        return True

    def grid(self) -> Iterator[dict]:
        """Every config, in a fixed order (the first knob varies slowest)."""
        names = list(self.knobs)
        for combo in itertools.product(*(self.knobs[n] for n in names)):
            yield dict(zip(names, combo))

    def random_config(self, rng: random.Random) -> dict:
        return {knob: rng.choice(values) for knob, values in self.knobs.items()}


def parse_search_space(data) -> SearchSpace:
    if not isinstance(data, dict):
        raise SearchSpaceError("search space must be a mapping with a `knobs` section")
    raw_knobs = data.get("knobs", data if "baseline" not in data else None)
    if not isinstance(raw_knobs, dict) or not raw_knobs:
        raise SearchSpaceError("search space needs a non-empty `knobs` mapping")

    knobs: dict[str, list] = {}
    for knob, spec in raw_knobs.items():
        if isinstance(spec, dict):
            values = _expand_range(knob, spec)
        elif isinstance(spec, list):
            values = list(spec)
        else:
            values = [spec]
        if not values:
            raise SearchSpaceError(f"knob {knob!r} has no values")
        knobs[str(knob)] = values

    baseline_spec = data.get("baseline") or {}
    if not isinstance(baseline_spec, dict):
        raise SearchSpaceError("`baseline` must be a mapping of knob: value")
    unknown = set(baseline_spec) - set(knobs)
    if unknown:
        raise SearchSpaceError(f"baseline sets unknown knob(s): {sorted(unknown)}")
    space = SearchSpace(knobs=knobs, baseline={})
    baseline = {knob: baseline_spec.get(knob, values[0]) for knob, values in knobs.items()}
    space.baseline = space.normalize(baseline)
    return space


def load_search_space(path: str | Path) -> SearchSpace:
    path = Path(path)
    if not path.is_file():
        raise SearchSpaceError(f"search space file not found: {path}")
    try:
        data = yaml_io.load(path)
    except yaml_io.YAMLError as exc:
        raise SearchSpaceError(f"could not parse {path}: {exc}") from exc
    return parse_search_space(data)
