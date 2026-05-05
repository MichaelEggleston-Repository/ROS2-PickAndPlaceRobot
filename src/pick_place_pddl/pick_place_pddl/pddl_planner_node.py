"""
pddl_planner_node.py
====================
ROS2 service node that solves a blocks-world PDDL problem and returns an
ordered list of PddlAction steps.

Service:  /generate_plan  (pick_place_interfaces/srv/GeneratePlan)

The node uses the ``unified-planning`` library with the ``pyperplan`` engine to
solve the problem entirely in-process.  No separate planner binary is needed.

Domain summary
--------------
Types   : cube, slot
Objects : red_cube, green_cube, blue_cube, slot_1, slot_2, slot_3

Predicates
  on_surface(cube, slot)   cube is resting in a conveyor slot
  on_cube(top, bottom)     top cube rests directly on bottom cube
  clear(cube)              nothing is on top of this cube
  slot_clear(slot)         slot contains no cube
  holding(cube)            gripper is holding this cube
  gripper_empty            gripper holds nothing

Actions
  pick_from_surface(cube, slot)
  pick_from_stack(top, bottom)
  place_on_surface(cube, slot)
  place_on_cube(top, bottom)

Goal types
  "stack"   : build a vertical stack; goal_sequence = [bottom, ..., top]
  "arrange" : place each cube into the matching slot in order;
              goal_sequence[0] → slot_1, [1] → slot_2, [2] → slot_3
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from pick_place_interfaces.srv import GeneratePlan
from pick_place_interfaces.msg import PddlAction

# unified-planning imports — imported lazily so the node starts cleanly even
# if the library is missing (it will fail gracefully on the first service call).
try:
    from unified_planning.shortcuts import (
        BoolType,
        Fluent,
        InstantaneousAction,
        Object,
        OneshotPlanner,
        Problem,
        UserType,
    )
    _UP_AVAILABLE = True
except ImportError:
    _UP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------

class PddlPlannerNode(Node):

    _SERVICE_NAME = "/generate_plan"

    # Known cube ids (colour encoded in the id).
    _CUBE_IDS = {"red_cube", "green_cube", "blue_cube"}

    # Slot names used for arrange goals (fixed order).
    _SLOT_NAMES = ["slot_1", "slot_2", "slot_3"]

    def __init__(self) -> None:
        super().__init__("pddl_planner_node")

        if not _UP_AVAILABLE:
            self.get_logger().error(
                "unified-planning library is not installed. "
                "Install it with:  pip install unified-planning pyperplan"
            )

        self._srv = self.create_service(
            GeneratePlan,
            self._SERVICE_NAME,
            self._handle_generate_plan,
        )
        self.get_logger().info(
            f"PDDL planner node ready, serving '{self._SERVICE_NAME}'."
        )

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_generate_plan(
        self,
        request: GeneratePlan.Request,
        response: GeneratePlan.Response,
    ) -> GeneratePlan.Response:

        self.get_logger().info(
            f"GeneratePlan request received — goal_type='{request.goal_type}' "
            f"goal_sequence={list(request.goal_sequence)}"
        )

        if not _UP_AVAILABLE:
            response.success = False
            response.message = "unified-planning library is not available."
            return response

        try:
            plan = self._solve(request)
        except Exception as exc:  # pylint: disable=broad-except
            msg = repr(exc) if not str(exc) else str(exc)
            self.get_logger().error(f"Planning failed with exception: {msg}")
            response.success = False
            response.message = msg
            return response

        if plan is None:
            response.success = False
            response.message = "Planner found no solution for the given goal."
            return response

        response.success = True
        response.message = f"Plan found with {len(plan)} steps."
        response.plan = plan
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    # PDDL problem builder and solver
    # ------------------------------------------------------------------

    def _solve(self, request: GeneratePlan.Request) -> list[PddlAction] | None:
        """
        Build a unified-planning Problem from the request and solve it.

        Returns a list of PddlAction messages on success, None if no plan
        could be found.
        """
        # ---- Types -------------------------------------------------------
        cube_type = UserType("cube")
        slot_type = UserType("slot")

        # ---- Fluents (predicates) ----------------------------------------
        on_surface    = Fluent("on_surface",    BoolType(), cube=cube_type, slot=slot_type)
        on_cube_f     = Fluent("on_cube",       BoolType(), top=cube_type, bottom=cube_type)
        clear         = Fluent("clear",         BoolType(), cube=cube_type)
        slot_clear    = Fluent("slot_clear",    BoolType(), slot=slot_type)
        holding       = Fluent("holding",       BoolType(), cube=cube_type)
        gripper_empty = Fluent("gripper_empty", BoolType())

        # ---- Objects -------------------------------------------------------
        # Only create objects for cubes actually present in the detection.
        present_cube_ids = {obj.id for obj in request.detected_objects
                            if obj.id in self._CUBE_IDS}

        cube_objs: dict[str, Object] = {
            cid: Object(cid, cube_type) for cid in present_cube_ids
        }
        slot_objs: dict[str, Object] = {
            sn: Object(sn, slot_type) for sn in self._SLOT_NAMES
        }

        # ---- Actions -------------------------------------------------------

        # pick_from_surface(c: cube, s: slot)
        pick_from_surface = InstantaneousAction("pick_from_surface",
                                                c=cube_type, s=slot_type)
        c = pick_from_surface.parameter("c")
        s = pick_from_surface.parameter("s")
        pick_from_surface.add_precondition(on_surface(c, s))
        pick_from_surface.add_precondition(clear(c))
        pick_from_surface.add_precondition(gripper_empty())
        pick_from_surface.add_effect(on_surface(c, s), False)
        pick_from_surface.add_effect(slot_clear(s), True)
        pick_from_surface.add_effect(holding(c), True)
        pick_from_surface.add_effect(gripper_empty(), False)
        pick_from_surface.add_effect(clear(c), False)

        # pick_from_stack(t: cube, b: cube)
        pick_from_stack = InstantaneousAction("pick_from_stack",
                                              t=cube_type, b=cube_type)
        t = pick_from_stack.parameter("t")
        b = pick_from_stack.parameter("b")
        pick_from_stack.add_precondition(on_cube_f(t, b))
        pick_from_stack.add_precondition(clear(t))
        pick_from_stack.add_precondition(gripper_empty())
        pick_from_stack.add_effect(on_cube_f(t, b), False)
        pick_from_stack.add_effect(clear(b), True)
        pick_from_stack.add_effect(holding(t), True)
        pick_from_stack.add_effect(gripper_empty(), False)
        pick_from_stack.add_effect(clear(t), False)

        # place_on_surface(c: cube, s: slot)
        place_on_surface = InstantaneousAction("place_on_surface",
                                               c=cube_type, s=slot_type)
        c = place_on_surface.parameter("c")
        s = place_on_surface.parameter("s")
        place_on_surface.add_precondition(holding(c))
        place_on_surface.add_precondition(slot_clear(s))
        place_on_surface.add_effect(holding(c), False)
        place_on_surface.add_effect(gripper_empty(), True)
        place_on_surface.add_effect(on_surface(c, s), True)
        place_on_surface.add_effect(slot_clear(s), False)
        place_on_surface.add_effect(clear(c), True)

        # place_on_cube(t: cube, b: cube)
        place_on_cube = InstantaneousAction("place_on_cube",
                                            t=cube_type, b=cube_type)
        t = place_on_cube.parameter("t")
        b = place_on_cube.parameter("b")
        place_on_cube.add_precondition(holding(t))
        place_on_cube.add_precondition(clear(b))
        place_on_cube.add_effect(holding(t), False)
        place_on_cube.add_effect(gripper_empty(), True)
        place_on_cube.add_effect(on_cube_f(t, b), True)
        place_on_cube.add_effect(clear(b), False)
        place_on_cube.add_effect(clear(t), True)

        # ---- Problem -------------------------------------------------------
        problem = Problem("pick_and_place")

        for fluent in (on_surface, on_cube_f, clear, slot_clear,
                       holding, gripper_empty):
            problem.add_fluent(fluent, default_initial_value=False)

        for obj in list(cube_objs.values()) + list(slot_objs.values()):
            problem.add_object(obj)

        for action in (pick_from_surface, pick_from_stack,
                       place_on_surface, place_on_cube):
            problem.add_action(action)

        # ---- Initial state ------------------------------------------------
        # All slots start clear; gripper starts empty.
        for slot_obj in slot_objs.values():
            problem.set_initial_value(slot_clear(slot_obj), True)
        problem.set_initial_value(gripper_empty(), True)

        # Map cube_id → slot_name from the parallel request lists.
        slot_map: dict[str, str] = {}
        for i, obj in enumerate(request.detected_objects):
            if i < len(request.occupied_slots):
                slot_map[obj.id] = request.occupied_slots[i]

        cubes_on_surface = {
            cid for cid, sn in slot_map.items() if sn and cid in cube_objs
        }
        cubes_not_slotted = present_cube_ids - cubes_on_surface

        for cube_id, slot_name in slot_map.items():
            if cube_id not in cube_objs:
                continue
            if slot_name and slot_name in slot_objs:
                problem.set_initial_value(
                    on_surface(cube_objs[cube_id], slot_objs[slot_name]), True
                )
                problem.set_initial_value(slot_clear(slot_objs[slot_name]), False)
                problem.set_initial_value(clear(cube_objs[cube_id]), True)

        # Assign unslotted cubes to spare slots so the planner can reason
        # about them (3 cubes + 3 slots means this always resolves cleanly).
        used_slots = set(
            sn for sn in slot_map.values() if sn and sn in slot_objs
        )
        spare_slots = [sn for sn in self._SLOT_NAMES if sn not in used_slots]

        for cube_id in sorted(cubes_not_slotted):
            if cube_id not in cube_objs:
                continue
            if spare_slots:
                tmp_slot = spare_slots.pop(0)
                problem.set_initial_value(
                    on_surface(cube_objs[cube_id], slot_objs[tmp_slot]), True
                )
                problem.set_initial_value(slot_clear(slot_objs[tmp_slot]), False)
                problem.set_initial_value(clear(cube_objs[cube_id]), True)
            else:
                # Fallback — cube not in any slot (shouldn't occur with 3+3).
                problem.set_initial_value(clear(cube_objs[cube_id]), True)

        # ---- Goal ---------------------------------------------------------
        goal_seq  = list(request.goal_sequence)
        goal_type = request.goal_type

        if goal_type == "stack":
            # goal_seq = [bottom, ..., top]
            if len(goal_seq) < 2:
                raise ValueError(
                    f"Stack goal requires at least 2 cubes, got {goal_seq}"
                )
            for i in range(1, len(goal_seq)):
                top_id = goal_seq[i]
                bot_id = goal_seq[i - 1]
                if top_id not in cube_objs or bot_id not in cube_objs:
                    raise ValueError(
                        f"Stack goal references unknown cube: "
                        f"top={top_id} bottom={bot_id}"
                    )
                problem.add_goal(on_cube_f(cube_objs[top_id], cube_objs[bot_id]))

        elif goal_type == "arrange":
            for i, cube_id in enumerate(goal_seq):
                slot_name = self._SLOT_NAMES[i]
                if cube_id not in cube_objs:
                    raise ValueError(
                        f"Arrange goal references unknown cube: {cube_id}"
                    )
                if slot_name not in slot_objs:
                    raise ValueError(
                        f"Arrange goal references unknown slot: {slot_name}"
                    )
                problem.add_goal(
                    on_surface(cube_objs[cube_id], slot_objs[slot_name])
                )
        else:
            raise ValueError(f"Unknown goal_type: '{goal_type}'")

        # ---- Solve --------------------------------------------------------
        # Let unified-planning auto-select the best available engine for
        # the problem kind rather than hard-coding a planner name.
        with OneshotPlanner(problem_kind=problem.kind) as planner:
            result = planner.solve(problem)

        if result.plan is None:
            return None

        # ---- Translate plan to ROS messages --------------------------------
        ros_plan: list[PddlAction] = []
        for action_instance in result.plan.actions:
            act_name = action_instance.action.name
            # actual_parameters is a tuple of FNode (Object) values.
            params = [str(p) for p in action_instance.actual_parameters]

            msg = PddlAction()
            msg.action_type = act_name

            if act_name in ("pick_from_surface", "place_on_surface"):
                # params: [cube, slot]
                msg.cube_id  = params[0]
                msg.location = params[1]

            elif act_name in ("pick_from_stack", "place_on_cube"):
                # params: [top_cube, bottom_cube]
                msg.cube_id  = params[0]
                msg.location = params[1]

            else:
                raise ValueError(f"Unexpected action in plan: '{act_name}'")

            ros_plan.append(msg)
            self.get_logger().info(
                f"  Plan step: {act_name}({msg.cube_id}, {msg.location})"
            )

        return ros_plan


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = PddlPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
