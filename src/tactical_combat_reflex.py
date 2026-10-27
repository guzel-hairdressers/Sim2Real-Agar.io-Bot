"""Compatibility entry point for the corrected observable tactical filter.

The historical implementation is frozen under benchmarks for before/after
comparisons. This entry point no longer replaces neural foraging or split
choices with unconditional nutrient rules.
"""
from src.arena_tactics import safety_filter


def apply_tactical_combat_reflex(raw_action, obs):
    return safety_filter(raw_action, obs)
