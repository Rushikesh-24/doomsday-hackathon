"""Canonical facility taxonomy and prompt <-> class mapping."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CATEGORIES = {"facility", "navigation", "signage", "obstacle", "auxiliary"}


@dataclass(frozen=True)
class FacilityClass:
    name: str
    category: str
    prompts: tuple[str, ...]
    coco_labels: tuple[str, ...] = ()
    color: tuple[int, int, int] = (255, 255, 255)  # RGB
    threshold_scale: float = 1.0
    merge_fragments: bool = False


@dataclass
class Taxonomy:
    classes: dict[str, FacilityClass]
    _prompt_to_class: dict[str, str] = field(default_factory=dict)
    _coco_to_class: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for c in self.classes.values():
            for p in c.prompts:
                key = p.lower().strip()
                if key in self._prompt_to_class and self._prompt_to_class[key] != c.name:
                    raise ValueError(f"prompt '{p}' assigned to two classes")
                self._prompt_to_class[key] = c.name
            for lbl in c.coco_labels:
                self._coco_to_class[lbl.lower()] = c.name

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Taxonomy":
        data = yaml.safe_load(Path(path).read_text())
        classes = {}
        for name, spec in data["classes"].items():
            cat = spec.get("category", "facility")
            if cat not in CATEGORIES:
                raise ValueError(f"class {name}: unknown category {cat}")
            classes[name] = FacilityClass(
                name=name,
                category=cat,
                prompts=tuple(spec.get("prompts", [])),
                coco_labels=tuple(spec.get("coco_labels", [])),
                color=tuple(spec.get("color", [255, 255, 255])),
                threshold_scale=float(spec.get("threshold_scale", 1.0)),
                merge_fragments=bool(spec.get("merge_fragments", False)),
            )
        return cls(classes=classes)

    # ---- queries -------------------------------------------------------
    def all_prompts(self, only: list[str] | None = None) -> list[str]:
        out = []
        for c in self.classes.values():
            if only and c.name not in only:
                continue
            out.extend(c.prompts)
        return out

    def class_for_prompt(self, prompt: str) -> str | None:
        return self._prompt_to_class.get(prompt.lower().strip())

    def class_for_coco(self, label: str) -> str | None:
        return self._coco_to_class.get(label.lower())

    def category(self, class_name: str) -> str:
        return self.classes[class_name].category

    def color(self, class_name: str) -> tuple[int, int, int]:
        c = self.classes.get(class_name)
        return c.color if c else (255, 255, 255)

    def names(self) -> list[str]:
        return list(self.classes)


@lru_cache(maxsize=4)
def load_taxonomy(path: str) -> Taxonomy:
    return Taxonomy.from_yaml(path)
