"""Backward-compatible import for the renamed grounding module."""

from sim.grounding import ResolvedWaypoint, resolve_waypoint

__all__ = ("ResolvedWaypoint", "resolve_waypoint")
