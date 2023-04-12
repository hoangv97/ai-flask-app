import os
import json
import random


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def is_dev_mode():
    return os.getenv("DEV_MODE", "false").lower() == "true"


protected_data = os.getenv("PROTECTED_DATA", "").split(",")


def encode_protected_output(input: str):
    for i, data in enumerate(protected_data):
        input = input.replace(data, "PROTECTEDDATA_{}".format(i))
    return input


def decode_protected_output(input: str):
    for i, data in enumerate(protected_data):
        input = input.replace("PROTECTEDDATA_{}".format(i), data)
    return input


def parse_json_string(input: str) -> str:
    """Normalize a JSON string."""
    if input.startswith("```"):
        if input.startswith("```json"):
            input = input.replace("```json", "")
        input = input.replace("```", "")
    input = input.strip()
    return json.loads(input)


def random_number(min: int, max: int):
    return random.randint(min, max)
