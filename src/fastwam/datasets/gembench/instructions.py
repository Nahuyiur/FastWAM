from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


OFFICIAL_TRAIN_TASKVARS = [
    "push_button+0",
    "push_button+3",
    "push_button+4",
    "close_fridge+0",
    "close_laptop_lid+0",
    "close_microwave+0",
    "open_door+0",
    "open_box+0",
    "open_drawer+0",
    "open_drawer+2",
    "pick_and_lift+0",
    "pick_and_lift+2",
    "pick_and_lift+7",
    "pick_up_cup+8",
    "pick_up_cup+9",
    "pick_up_cup+11",
    "stack_blocks+30",
    "stack_blocks+36",
    "stack_blocks+39",
    "put_groceries_in_cupboard+0",
    "put_groceries_in_cupboard+3",
    "put_money_in_safe+0",
    "put_money_in_safe+1",
    "slide_block_to_color_target_peract+0",
    "slide_block_to_color_target_peract+1",
    "reach_and_drag_peract+14",
    "reach_and_drag_peract+18",
    "close_jar_peract+15",
    "close_jar_peract+16",
    "light_bulb_in_peract+17",
    "light_bulb_in_peract+19",
]

DEFAULT_TASKVAR_INSTRUCTIONS = {
    "push_button+0": ["push the maroon button"],
    "push_button+3": ["push the navy button"],
    "push_button+4": ["push the yellow button"],
    "close_fridge+0": ["close fridge"],
    "close_laptop_lid+0": ["close laptop lid"],
    "close_microwave+0": ["close microwave"],
    "open_door+0": ["open the door"],
    "open_box+0": ["open box"],
    "open_drawer+0": ["open bottom drawer"],
    "open_drawer+2": ["open top drawer"],
    "pick_and_lift+0": ["pick up the red block and lift it up to the target"],
    "pick_and_lift+2": ["pick up the lime block and lift it up to the target"],
    "pick_and_lift+7": ["pick up the cyan block and lift it up to the target"],
    "pick_up_cup+8": ["pick up the magenta cup"],
    "pick_up_cup+9": ["pick up the silver cup"],
    "pick_up_cup+11": ["pick up the orange cup"],
    "stack_blocks+30": ["stack 2 gray blocks"],
    "stack_blocks+36": ["stack 2 olive blocks"],
    "stack_blocks+39": ["stack 2 purple blocks"],
    "put_groceries_in_cupboard+0": ["put the crackers box in the cupboard"],
    "put_groceries_in_cupboard+3": ["put the soup can in the cupboard"],
    "put_money_in_safe+0": ["put the money away in the safe on the bottom shelf"],
    "put_money_in_safe+1": ["put the money away in the safe on the middle shelf"],
    "slide_block_to_color_target_peract+0": ["slide the block to green target"],
    "slide_block_to_color_target_peract+1": ["slide the block to blue target"],
    "reach_and_drag_peract+14": ["use the stick to drag the cube onto the teal target"],
    "reach_and_drag_peract+18": ["use the stick to drag the cube onto the black target"],
    "close_jar_peract+15": ["close the azure jar"],
    "close_jar_peract+16": ["close the violet jar"],
    "light_bulb_in_peract+17": ["screw in the rose light bulb"],
    "light_bulb_in_peract+19": ["screw in the white light bulb"],
}


def _normalize_instruction_payload(payload: object) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        raise ValueError(f"Instruction JSON must be a dict, got {type(payload)}")
    out: dict[str, list[str]] = {}
    for key, value in payload.items():
        taskvar = str(key)
        if isinstance(value, str):
            instructions = [value]
        elif isinstance(value, list):
            instructions = [str(v) for v in value if str(v).strip()]
        else:
            raise ValueError(f"Instruction entry for {taskvar!r} must be str or list[str].")
        if instructions:
            out[taskvar] = instructions
    return out


def load_instruction_map(instruction_json_path: str | None = None) -> dict[str, list[str]]:
    if instruction_json_path:
        path = Path(instruction_json_path).expanduser()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                loaded = _normalize_instruction_payload(json.load(f))
            merged = dict(DEFAULT_TASKVAR_INSTRUCTIONS)
            merged.update(loaded)
            return merged
    return dict(DEFAULT_TASKVAR_INSTRUCTIONS)


def taskvar_to_task_name(taskvar: str) -> str:
    return taskvar.split("+", 1)[0]


def fallback_instruction(taskvar: str) -> str:
    return taskvar_to_task_name(taskvar).replace("_peract", "").replace("_", " ")


def instruction_for_taskvar(
    taskvar: str,
    instruction_map: Mapping[str, list[str]] | None = None,
    instruction_index: int = 0,
) -> str:
    mapping = instruction_map or DEFAULT_TASKVAR_INSTRUCTIONS
    candidates = mapping.get(taskvar)
    if not candidates:
        return fallback_instruction(taskvar)
    return candidates[int(instruction_index) % len(candidates)]


def resolve_taskvars(taskvars: Iterable[str] | str | None) -> list[str] | None:
    if taskvars is None:
        return None
    if isinstance(taskvars, str):
        if taskvars in {"official_train", "train"}:
            return list(OFFICIAL_TRAIN_TASKVARS)
        return [item.strip() for item in taskvars.split(",") if item.strip()]
    return [str(item) for item in taskvars]


def iter_instructions(
    taskvars: Iterable[str],
    instruction_map: Mapping[str, list[str]] | None = None,
) -> list[str]:
    mapping = instruction_map or DEFAULT_TASKVAR_INSTRUCTIONS
    out: list[str] = []
    seen = set()
    for taskvar in taskvars:
        candidates = mapping.get(taskvar) or [fallback_instruction(taskvar)]
        for instruction in candidates:
            if instruction not in seen:
                seen.add(instruction)
                out.append(instruction)
    return out
