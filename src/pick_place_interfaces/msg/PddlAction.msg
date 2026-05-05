# PddlAction.msg
# A single action step produced by the PDDL planner.
#
# action_type values:
#   pick_from_surface  - pick cube from a conveyor slot
#   pick_from_stack    - pick cube off the top of another cube
#   place_on_surface   - place cube into a conveyor slot
#   place_on_cube      - place cube on top of another cube
#
# location:
#   For surface actions: slot name, e.g. "slot_1"
#   For stack actions:   cube id of the support cube, e.g. "green_cube"

string action_type
string cube_id
string location
