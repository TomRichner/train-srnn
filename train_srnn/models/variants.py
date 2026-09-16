"""SRNN variant names.

A variant name is ``srnn`` followed by tokens, each switching one ablation
relative to the shared model config, and an optional ``-seed<n>`` suffix
selecting the recurrent-matrix seed::

    srnn                       full model
    srnn-no-adapt              no SFA, no STD
    srnn-e-only-skip           adaptation on E neurons only, skip connection
    srnn-no-dales-skip-seed3   unsigned weights, skip, recurrent seed 3

Names are lower-case and canonicalised to the token order below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

PREFIX = "srnn"
SEED_SEP = "-seed"

# Canonical order. Each token overrides the listed flags.
TOKENS: dict[str, dict[str, Any]] = {
    "sfa1-std1": dict(n_a_E=1, n_a_I=1, n_b_E=1, n_b_I=1),
    "sfa3-std2": dict(n_a_E=3, n_a_I=3, n_b_E=2, n_b_I=2),
    "no-adapt": dict(n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0),
    "sfa-only": dict(n_b_E=0, n_b_I=0),
    "std-only": dict(n_a_E=0, n_a_I=0),
    "e-only": dict(n_a_I=0, n_b_I=0),
    "no-dales": dict(dales=False),
    "per-neuron": dict(per_neuron=True),
    "echo": dict(echo=True),
    "skip": dict(skip=True),
}

# Short spellings for common token pairs; parsed as the pair and printed as the alias.
ALIASES: dict[str, tuple[str, ...]] = {
    "sfa-e-only": ("sfa-only", "e-only"),
    "std-e-only": ("std-only", "e-only"),
}

FLAGS = ("dales", "n_a_E", "n_a_I", "n_b_E", "n_b_I", "per_neuron", "echo", "skip")


@dataclass(frozen=True)
class SRNNVariant:
    name: str                 # canonical name, with the seed suffix only when given explicitly
    tokens: tuple[str, ...]
    seed: int
    dales: bool
    n_a_E: int
    n_a_I: int
    n_b_E: int
    n_b_I: int
    per_neuron: bool
    echo: bool
    skip: bool

    @property
    def base_name(self) -> str:
        return canonical_name(self.tokens)


def canonical_name(tokens: tuple[str, ...] | list[str], seed: Optional[int] = None) -> str:
    ordered = [t for t in TOKENS if t in tokens]
    for alias, pair in ALIASES.items():
        if all(t in ordered for t in pair):
            ordered[ordered.index(pair[0])] = alias
            ordered = [t for t in ordered if t not in pair[1:]]
    name = "-".join([PREFIX, *ordered])
    return f"{name}{SEED_SEP}{seed}" if seed is not None else name


def parse_name(name: str) -> tuple[tuple[str, ...], Optional[int]]:
    """Split a name into its tokens and explicit seed (or None)."""
    raw = name.strip().lower()
    seed: Optional[int] = None
    if SEED_SEP in raw:
        raw, _, seed_str = raw.rpartition(SEED_SEP)
        try:
            seed = int(seed_str)
        except ValueError:
            raise ValueError(f"Malformed seed suffix in variant name {name!r}") from None
    if raw != PREFIX and not raw.startswith(PREFIX + "-"):
        raise ValueError(f"Variant name {name!r} must start with {PREFIX!r}")
    rest = raw[len(PREFIX):].lstrip("-")
    tokens: list[str] = []
    vocab = {**{t: (t,) for t in TOKENS}, **ALIASES}
    while rest:
        for tok in sorted(vocab, key=len, reverse=True):
            if rest == tok or rest.startswith(tok + "-"):
                for t in vocab[tok]:
                    if t in tokens:
                        raise ValueError(f"Duplicate token {t!r} in variant name {name!r}")
                    tokens.append(t)
                rest = rest[len(tok):].lstrip("-")
                break
        else:
            raise ValueError(
                f"Unknown token at {rest!r} in variant name {name!r}. "
                f"Known tokens: {', '.join(TOKENS)}")
    return tuple(tokens), seed


def make_variant(name: str, base: Mapping[str, Any], default_seed: int) -> SRNNVariant:
    """Resolve a name against the shared flags in *base* (an SRNNModelConfig-like mapping)."""
    tokens, seed = parse_name(name)
    conditions = {"no-adapt", "sfa1-std1", "sfa3-std2"}.intersection(tokens)
    if len(conditions) > 1 or (conditions and {"sfa-only", "std-only", "e-only"}.intersection(tokens)):
        raise ValueError("Conflicting adaptation condition tokens")
    if {"sfa-only", "std-only"}.issubset(tokens):
        raise ValueError("Conflicting sfa-only and std-only tokens")
    flags = {k: base[k] for k in FLAGS}
    for tok in tokens:
        flags.update(TOKENS[tok])
    return SRNNVariant(
        name=canonical_name(tokens, seed),
        tokens=tokens,
        seed=default_seed if seed is None else seed,
        **flags,
    )


def expand(names: list[str], seeds: Optional[list[int]], base: Mapping[str, Any],
           default_seed: int) -> list[SRNNVariant]:
    """Cross names with seeds, variant-major: ``([a, b], [1, 2]) -> a-seed1, a-seed2, b-seed1, b-seed2``.

    Without *seeds* the names pass through with ``default_seed``. Variants
    that share a seed share one recurrent matrix, so comparisons are paired.
    """
    if not seeds:
        return [make_variant(n, base, default_seed) for n in names]
    out = []
    for n in names:
        if SEED_SEP in n.lower():
            raise ValueError(
                f"variant_seeds was given, but {n!r} already carries a seed suffix")
        for s in seeds:
            out.append(make_variant(f"{n}{SEED_SEP}{int(s)}", base, default_seed))
    return out
