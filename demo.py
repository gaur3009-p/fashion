import gradio as gr
import requests
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

repo = "artificialguybr/TshirtDesignRedmond-V2"
def infer(color_prompt, dress_type_prompt, design_prompt, text):
    prompt = (
        f"A single {color_prompt} colored {dress_type_prompt} featuring a bold {design_prompt} design printed on the {dress_type_prompt}, hanging on a plain wall. The soft light and shadows, creating a striking contrast against the minimal background, evoking modern sophistication.")
    full_prompt = f"{prompt}"

    print("Generating image with prompt:", full_prompt)
    api_url = f"https://api-inference.huggingface.co/models/{repo}"
    #token = os.getenv("API_TOKEN")  # Uncomment and use your Hugging Face API token
    headers = {
        #"Authorization": f"Bearer {token}"
    }
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "negative_prompt": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3), (Poor Fit, ill-fitting, awkward proportions, baggy where it shouldn't be, tight in wrong places, Bad Texture, low-quality fabric, wrinkled, rough texture, pixelated details, artificial shine, Cluttered Design, overly busy, too many patterns, excessive contrast, distracting elements, Outdated Fashion, old-fashioned, style(unless intentional), outdated trends, dull colors, Bad Composition, misaligned prints, asymmetrical in an unintentional way, weird placement of logos, Cheap Look, plastic-like fabric, low-quality print, faded colors, generic fast fashion, Unrealistic Details, floating textures, distorted logos, unnatural fabric folds, Unwanted Features, holes, rips, stains, unfinished seams, torn edges)",
            "num_inference_steps": 30,
            "scheduler": "DPMSolverMultistepScheduler"
        },
    }

    error_count = 0
    pbar = tqdm(total=None, desc="Loading model")
    while True:
        print("Sending request to API...")
        response = requests.post(api_url, headers=headers, json=payload)
        print("API response status code:", response.status_code)
        if response.status_code == 200:
            print("Image generation successful!")
            return Image.open(BytesIO(response.content))
        elif response.status_code == 503:
            time.sleep(1)
            pbar.update(1)
        elif response.status_code == 500 and error_count < 5:
            time.sleep(1)
            error_count += 1
        else:
            print("API Error:", response.status_code)
            raise Exception(f"API Error: {response.status_code}")

# Gradio Interface
iface = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(lines=1, placeholder="Color Prompt"),         # color_prompt
        gr.Textbox(lines=1, placeholder="Dress Type Prompt"),    # dress_type_prompt
        gr.Textbox(lines=2, placeholder="Design Prompt"),        # design_prompt
        gr.Textbox(lines=1, placeholder="Text (Optional)"),      # text
    ],
    outputs="image",
    title="Make your Brand",
    description="Generation of clothes",
    examples=[["Red", "T-shirt", "Simple design", "Stylish Text"]]
)

print("Launching Gradio interface...")
iface.launch() 