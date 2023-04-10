import os


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
