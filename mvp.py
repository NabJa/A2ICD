import gradio as gr
import requests
import torch
import torchaudio
import yaml
from openai import OpenAI
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def get_api_key(source="deepseek"):
    with open("api_keys.yml") as stream:
        api_keys = yaml.safe_load(stream)
        return api_keys[source]


whisper_id = "openai/whisper-large-v2"  # "openai/whisper-base"

client = OpenAI(
    api_key=get_api_key("deepseek"),
    base_url="https://api.deepseek.com",
)

audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
audio_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")


def check_icd_link(code):
    # Construct the URL for checking the existence of the ICD code
    url = f"https://icd.who.int/browse10/2019/en/JsonGetParentConceptIDsToRoot?ConceptId={code}"
    response = requests.get(url, verify=False)

    new_code = code

    # Check if the response is valid and not null or empty
    if response.status_code == 200:
        data = response.json()
        i = len(code) - 1
        while not data:
            new_code = code[:i]
            url = f"https://icd.who.int/browse10/2019/en/JsonGetParentConceptIDsToRoot?ConceptId={new_code}"
            response = requests.get(url, verify=False)
            data = response.json()
            i -= 1
            if i == 0:
                break

    return new_code, code


def predict_icd(text):
    messages = [
        {
            "role": "system",
            "content": "You are an expert medical assistant. Extract the most relevant ICD-10 code(s) for the following condition. Only return the five most relevant ICD-10 codes, nothing else.",
        },
        {"role": "user", "content": f"Condition: {text}"},
        {"role": "system", "content": "Comma-separated 5 most relevant ICD-10 codes:"},
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
        stream=False,
    )

    # Extract the ICD-10 codes from the response
    icd_codes = response.choices[0].message.content.split(", ")

    # Make sure its only five codes
    icd_codes = icd_codes[:5]

    # Make sure the ICD-10 codes are valid links
    valid_codes = [check_icd_link(c) for c in icd_codes]

    icd_markdown = f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #1f2937;">
        <strong>ICD-10 Codes:</strong>
        <ul>
            {"".join([f'<li><a href="https://icd.who.int/browse10/2019/en#/{code_link}" target="_blank">{code_name}</a></li>' for code_link, code_name in valid_codes])}
        </ul>
    </div>
    """

    return icd_markdown


def transcribe(audio, language="en-en"):
    if language == "de-de":
        forced_decoder_ids = audio_processor.get_decoder_prompt_ids(
            task="transcribe", language="german"
        )
    elif language == "de-en":
        forced_decoder_ids = audio_processor.get_decoder_prompt_ids(
            task="translate", language="english"
        )
    elif language == "en-en":
        forced_decoder_ids = audio_processor.get_decoder_prompt_ids(
            task="transcribe", language="english"
        )
    else:
        forced_decoder_ids = audio_processor.get_decoder_prompt_ids()

    sr, y = audio

    y = torch.tensor(y, dtype=torch.float32)

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(dim=1)

    # Resample to 16kHz if needed
    target_sr = 16_000
    if sr != target_sr:
        y = torchaudio.functional.resample(y, sr, target_sr)
        sr = target_sr

    y /= torch.max(torch.abs(y))

    inputs = audio_processor(y, sampling_rate=sr, return_tensors="pt")

    predicted_ids = audio_model.generate(
        inputs.input_features,
        forced_decoder_ids=forced_decoder_ids,
        pad_token_id=audio_processor.tokenizer.eos_token_id,
    )

    transcription = audio_processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )

    text = transcription[0]

    return text  # Return transcription


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            with gr.Column():
                audio_input = gr.Audio(sources="microphone", label="Input Audio")
                language_radio = gr.Radio(
                    ["de-de", "de-en", "en-en"], value="en-en", label="Language"
                )
                transcribe_btn = gr.Button("Transcribe")
            with gr.Column():
                transcribed_text = gr.Textbox(
                    label="Transcribed", interactive=True
                )  # Now editable!
                icd_btn = gr.Button("Get ICD-10 Codes")
            with gr.Column():
                icd10_codes = gr.Markdown(label="ICD-10 Code(s)")

        # Button for transcription
        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, language_radio],
            outputs=[transcribed_text],
        )

        # Button for ICD-10 prediction
        icd_btn.click(
            fn=predict_icd,
            inputs=[transcribed_text],
            outputs=[icd10_codes],
        )

    demo.launch(debug=False, share=False)
