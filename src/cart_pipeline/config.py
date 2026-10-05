from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

@dataclass(frozen=True)
class MechanismSpace:
    names: tuple[str, ...] = ('s_mig', 's_cyt', 's_exh', 's_esc', 's_env', 's_hyp')
    lower: float = 0.6
    upper: float = 1.4

    @property
    def dim(self) -> int:
        return len(self.names)

    def to_unit(self, x):
        return (x - self.lower) / (self.upper - self.lower)

    def from_unit(self, z):
        return self.lower + z * (self.upper - self.lower)
