"""
world_state_tracker.py
======================
Bridges the symbolic PDDL plan and the physical robot workspace.

Responsibilities
----------------
* Stores the current physical pose (x, y, z) of every cube tracked during a
  plan execution.
* Computes the TCP target for each PDDL action (pick / place coordinates),
  accounting for stack geometry so cubes placed on other cubes land at the
  correct height.
* Updates its internal bookkeeping as each PDDL action succeeds so that
  subsequent actions use up-to-date positions.

Coordinate conventions
----------------------
All positions are in the robot base frame (metres).
  x, y  — horizontal position of the cube centroid
  z      — height of the cube centroid above the table

The tracker is constructed with:
  cube_poses   : {cube_id: (x, y, z, half_height, half_width)}
  slot_poses   : {slot_name: (x, y, z_surface)}
                 z_surface is the height of the slot surface (table top), not a
                 centroid — the first cube placed there will sit at
                 z_surface + half_height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CubeState:
    """Physical state of one cube during plan execution."""

    cube_id: str

    # Current centroid position in robot base frame (metres).
    x: float
    y: float
    z: float

    # Sensor-derived geometry half-extents (metres).
    half_height: float
    half_width: float

    # Symbolic location bookkeeping (kept in sync with executed actions).
    on_surface: Optional[str] = None       # slot name, or None
    on_cube: Optional[str] = None          # cube_id below, or None
    supporting: Optional[str] = None       # cube_id above, or None


@dataclass
class SlotState:
    """Physical state of one conveyor slot."""

    slot_name: str
    x: float
    y: float
    z_surface: float          # height of the surface (table top) in metres
    occupied_by: Optional[str] = None     # cube_id currently in slot, or None


# ---------------------------------------------------------------------------
# WorldStateTracker
# ---------------------------------------------------------------------------

class WorldStateTracker:
    """
    Tracks cube and slot states through PDDL plan execution and translates
    symbolic actions into physical TCP targets.

    Parameters
    ----------
    cube_data : dict
        {cube_id: (x, y, z_centroid, half_height, half_width)}
        Populated from DetectedObject messages at the start of each plan.
    slot_data : dict
        {slot_name: (x, y, z_surface)}
        Fixed absolute positions configured as ROS2 parameters.
    occupied_slots : list of (slot_name, cube_id) tuples
        Slots that already contain a cube at the start of planning.
    """

    def __init__(
        self,
        cube_data: Dict[str, Tuple[float, float, float, float, float]],
        slot_data: Dict[str, Tuple[float, float, float]],
        occupied_slots: list[Tuple[str, str]] | None = None,
    ) -> None:
        # Build cube state table.
        self._cubes: Dict[str, CubeState] = {}
        for cube_id, (x, y, z, hh, hw) in cube_data.items():
            self._cubes[cube_id] = CubeState(
                cube_id=cube_id, x=x, y=y, z=z,
                half_height=hh, half_width=hw,
            )

        # Build slot state table.
        self._slots: Dict[str, SlotState] = {}
        for slot_name, (x, y, z_surf) in slot_data.items():
            self._slots[slot_name] = SlotState(
                slot_name=slot_name, x=x, y=y, z_surface=z_surf,
            )

        # Initialise symbolic/physical location for pre-occupied slots.
        for slot_name, cube_id in (occupied_slots or []):
            if slot_name in self._slots and cube_id in self._cubes:
                slot = self._slots[slot_name]
                cube = self._cubes[cube_id]
                slot.occupied_by = cube_id
                cube.on_surface = slot_name
                # Keep the detected centroid z as-is — it is derived from
                # the actual depth measurement and is the ground truth.

    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------

    def cube_state(self, cube_id: str) -> CubeState:
        """Return the CubeState for cube_id (raises KeyError if unknown)."""
        return self._cubes[cube_id]

    def slot_state(self, slot_name: str) -> SlotState:
        """Return the SlotState for slot_name (raises KeyError if unknown)."""
        return self._slots[slot_name]

    def top_of_stack(self, base_cube_id: str) -> str:
        """
        Walk the 'supporting' chain upward and return the id of the topmost
        cube in the stack that has base_cube_id at its base.
        """
        current = base_cube_id
        while self._cubes[current].supporting is not None:
            current = self._cubes[current].supporting
        return current

    # ------------------------------------------------------------------
    # TCP target computation
    # ------------------------------------------------------------------

    def pick_pose(self, cube_id: str) -> Tuple[float, float, float, float]:
        """
        Return (x, y, z_centroid, half_width) — the TCP grasp pose for the
        cube's current centroid and its estimated width for gripper tolerance.
        """
        c = self._cubes[cube_id]
        return c.x, c.y, c.z, c.half_width

    def place_pose_on_surface(
        self, cube_id: str, slot_name: str
    ) -> Tuple[float, float, float]:
        """
        Return (x, y, z_centroid) for placing cube_id into slot_name.
        The centroid z is slot.z_surface + cube.half_height.
        """
        slot = self._slots[slot_name]
        cube = self._cubes[cube_id]
        z = slot.z_surface + cube.half_height
        return slot.x, slot.y, z

    def place_pose_on_cube(
        self, top_cube_id: str, bottom_cube_id: str
    ) -> Tuple[float, float, float]:
        """
        Return (x, y, z_centroid) for placing top_cube_id on bottom_cube_id.

        The bottom cube must currently be the topmost cube in its stack.
        z_centroid = bottom_cube.z + bottom_cube.half_height + top_cube.half_height
        """
        bottom = self._cubes[bottom_cube_id]
        top = self._cubes[top_cube_id]
        z = bottom.z + bottom.half_height + top.half_height
        return bottom.x, bottom.y, z

    # ------------------------------------------------------------------
    # State update after successful action execution
    # ------------------------------------------------------------------

    def apply_pick_from_surface(self, cube_id: str, slot_name: str) -> None:
        """Update state after successfully picking cube_id from slot_name."""
        cube = self._cubes[cube_id]
        slot = self._slots[slot_name]

        cube.on_surface = None
        slot.occupied_by = None

    def apply_pick_from_stack(self, top_cube_id: str, bottom_cube_id: str) -> None:
        """Update state after successfully picking top_cube_id off bottom_cube_id."""
        top = self._cubes[top_cube_id]
        bottom = self._cubes[bottom_cube_id]

        top.on_cube = None
        bottom.supporting = None

    def apply_place_on_surface(self, cube_id: str, slot_name: str) -> None:
        """Update state after successfully placing cube_id into slot_name."""
        slot = self._slots[slot_name]
        cube = self._cubes[cube_id]

        # Update symbolic location.
        cube.on_surface = slot_name
        slot.occupied_by = cube_id

        # Update physical centroid.
        cube.x = slot.x
        cube.y = slot.y
        cube.z = slot.z_surface + cube.half_height

    def apply_place_on_cube(self, top_cube_id: str, bottom_cube_id: str) -> None:
        """Update state after successfully placing top_cube_id onto bottom_cube_id."""
        top = self._cubes[top_cube_id]
        bottom = self._cubes[bottom_cube_id]

        # Update symbolic location.
        top.on_cube = bottom_cube_id
        bottom.supporting = top_cube_id

        # Physical centroid: directly above the bottom cube.
        top.x = bottom.x
        top.y = bottom.y
        top.z = bottom.z + bottom.half_height + top.half_height

    # ------------------------------------------------------------------
    # Convenience: apply any PddlAction after execution
    # ------------------------------------------------------------------

    def apply_action(self, action_type: str, cube_id: str, location: str) -> None:
        """
        Apply the world-state update for a successfully executed PDDL action.

        Parameters
        ----------
        action_type : str
            One of: pick_from_surface, pick_from_stack,
                    place_on_surface, place_on_cube
        cube_id : str
            The cube that was acted upon.
        location : str
            Slot name (surface actions) or support cube id (stack actions).
        """
        if action_type == "pick_from_surface":
            self.apply_pick_from_surface(cube_id, location)
        elif action_type == "pick_from_stack":
            self.apply_pick_from_stack(cube_id, location)
        elif action_type == "place_on_surface":
            self.apply_place_on_surface(cube_id, location)
        elif action_type == "place_on_cube":
            self.apply_place_on_cube(cube_id, location)
        else:
            raise ValueError(f"Unknown PDDL action type: '{action_type}'")
