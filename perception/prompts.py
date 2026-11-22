"""Prompts for asking the vision model what the robot's cameras see."""

SYSTEM = (
    "You are the eyes of a dual-arm tabletop robot. You receive images from its cameras. "
    "Describe only what is visible; say when you are unsure. Do not identify or describe people."
)

DESCRIBE = (
    "List the objects on the table. For each: name, colour, and rough position in the image "
    "(left / centre / right, near / far). Then say which objects a robot gripper could pick up."
)


def scene_json_prompt() -> str:
    """Prompt asking for a machine-readable list of objects, for the dashboard and later planning.

    TODO(you): define the scene schema. What the robot later does with "what the camera sees"
    depends entirely on which fields you ask for here.
    """
    raise NotImplementedError("scene_json_prompt: write the schema prompt (see the TODO above)")
