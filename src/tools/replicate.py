import json
import traceback
from typing import Literal

from langchain.agents import tool
import replicate
from ..utils.helper import decode_protected_output

Output_Type = Literal[
    "image",
    "single_image",
    "text",
    "transcription",
    "audio",
]


def get_prediction(
    model_name: str, model_version: str, query: str, output_type: Output_Type
):
    try:
        query = decode_protected_output(query)

        # normalize query string
        if query.startswith("```"):
            if query.startswith("```json"):
                query = query.replace("```json", "")
            query = query.replace("```", "")
        query = query.strip()

        input = json.loads(query)
        print(input)

        model = replicate.models.get(model_name)
        version = model.versions.get(model_version)
        prediction = replicate.predictions.create(
            version=version,
            input=input,
        )
        prediction = json.loads(prediction.json())
        prediction["output_type"] = output_type
        del prediction["version"]

        return {
            "success": True,
            "prediction": prediction,
            "model": {
                "name": model_name,
                "version": model_version,
            },
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": e.args,
        }


@tool(
    "stable-diffusion",
    return_direct=True,
)
def stable_diffusion(query: str):
    """useful for when you need to create an image or imagine something. The input of this tool in json format only with prompt key as description of the image, num_outputs key as number of result in integer."""
    return get_prediction(
        model_name="stability-ai/stable-diffusion",
        model_version="db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        query=query,
        output_type="image",
    )


@tool(
    "openjourney",
    return_direct=True,
)
def openjourney(query: str):
    """useful for when you need to create an image or imagine something. The input of this tool in json format only with prompt key as description of the image, num_outputs key as number of result in integer."""
    return get_prediction(
        model_name="prompthero/openjourney",
        model_version="9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb",
        query=query,
        output_type="image",
    )


@tool(
    "blip-2",
    return_direct=True,
)
def blip_2(query: str):
    """useful for when you need to answer a question about an image URL. The input of this tool in json format only with image key as image's URL only, question key as user's question."""
    return get_prediction(
        model_name="andreasjansson/blip-2",
        model_version="4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
        query=query,
        output_type="text",
    )


@tool(
    "controlnet-hough",
    return_direct=True,
)
def controlnet_hough(query: str):
    """useful for when you need to modify/edit an image URL. The input of this tool in json format only with image key as image's URL only, prompt key as user's prompt, num_samples key as number of result in string."""
    return get_prediction(
        model_name="jagilley/controlnet-hough",
        model_version="854e8727697a057c525cdb45ab037f64ecca770a1769cc52287c2e56472a247b",
        query=query,
        output_type="image",
    )


@tool(
    "controlnet-scribble",
    return_direct=True,
)
def controlnet_scribble(query: str):
    """useful for when you need to modify/edit an scribbled drawings image URL to a new detailed images. The input of this tool in json format only with image key as image's URL only, prompt key as user's prompt, num_samples key as number of result in string."""
    return get_prediction(
        model_name="jagilley/controlnet-scribble",
        model_version="435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117",
        query=query,
        output_type="image",
    )


@tool(
    "instruct-pix2pix",
    return_direct=True,
)
def instruct_pix2pix(query: str):
    """useful for when you need to modify/edit an image URL with user instructions. The input of this tool in json format only with image key as image's URL only, prompt key as user's prompt, num_outputs key as number of result in integer."""
    return get_prediction(
        model_name="timothybrooks/instruct-pix2pix",
        model_version="30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f",
        query=query,
        output_type="image",
    )


@tool(
    "codeformer",
    return_direct=True,
)
def codeformer(query: str):
    """useful for when you need to restore/enhance an image URL. The input of this tool in json format only with image key as image's URL only."""
    return get_prediction(
        model_name="sczhou/codeformer",
        model_version="7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56",
        query=query,
        output_type="single_image",
    )


@tool(
    "audio-ldm",
    return_direct=True,
)
def audio_ldm(query: str):
    """useful for when you need to generate audio from user's instructions. The input of this tool in json format only with text key as user's prompt, duration key as duration of the audio in seconds in string format (if not mentioned set it to 5.0)."""
    return get_prediction(
        model_name="haoheliu/audio-ldm",
        model_version="b61392adecdd660326fc9cfc5398182437dbe5e97b5decfb36e1a36de68b5b95",
        query=query,
        output_type="audio",
    )


tools = [
    stable_diffusion,
    blip_2,
    controlnet_hough,
    openjourney,
    controlnet_scribble,
    instruct_pix2pix,
    codeformer,
    audio_ldm,
]
