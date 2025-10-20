import gradio as gr
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.string import StrOutputParser
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client for audio and transcription
openai_client = OpenAI()

# Initialize language model
llm = init_chat_model("openai:gpt-4.1") | StrOutputParser()


def transcribe(audio: str) -> str:
    """Transcribe the given audio file using OpenAI's Whisper model."""
    with open(audio, "rb") as file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", file=file
        )
        transcribed = transcript.text

    system_prompt = """
        You are an AI agent that corrects transciped medical text for any errors.
        Only return the corrected text, nothing else. If the text is correct, return it as is.
        E.g.:
            User: The patient has a history of diabetis and hypertenzion.
            Agent: The patient has a history of diabetes and hypertension.
        """

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(content=transcribed),
    ]
    response = llm.invoke(messages)

    return response


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


def predict_icd(text: str) -> str:
    """Predict ICD-10 codes from the given text using the language model."""

    system_prompt = """
        You are a AI agent that extracts the most relevant ICD-10 code(s) for a given condition.
        Only return the five most relevant ICD-10 codes as a comma-separated list, nothing else.

        E.g.:
            User: Hypertension and diabetes mellitus.
            Agent: I10, I11.9, I12.9, E10.9, E11.9
    """

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(content=text),
    ]

    response = llm.invoke(messages)

    # Extract the ICD-10 codes from the response
    # Make sure its only five codes
    icd_codes = response.split(", ")[:5]

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


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload or record audio")
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
            inputs=[audio_input],
            outputs=[transcribed_text],
        )

        # Button for ICD-10 prediction
        icd_btn.click(
            fn=predict_icd,
            inputs=[transcribed_text],
            outputs=[icd10_codes],
        )

    demo.launch(debug=False, share=False)
