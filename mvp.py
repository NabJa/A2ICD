from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.string import StrOutputParser
from openai import OpenAI

from create_vector_store import load_vector_store
from icd import format_top_k_icd_codes
from interface import run_interface

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


def predict_icd(text: str, vector_store: Chroma) -> str:
    """Predict ICD-10 codes from the given text using the language model."""

    system_prompt = """
        You are a AI agent that extracts the most relevant ICD-10 code(s) for a given condition.
        You are given some context containing ICD-10 codes and their descriptions.
        Only return the five most relevant ICD-10 codes as a comma-separated list, nothing else.

        E.g.:
            User: Hypertension and diabetes mellitus.
            Agent: I10, I11.9, I12.9, E10.9, E11.9

        Context:
        {context}
    """

    context = vector_store.similarity_search(text, k=10)
    context = "\n".join([doc.page_content for doc in context])
    system_prompt = system_prompt.format(context=context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text),
    ]

    response = llm.invoke(messages)

    return format_top_k_icd_codes(response, k=5)


if __name__ == "__main__":
    from functools import partial

    vector_store = load_vector_store()

    predict_icd = partial(predict_icd, vector_store=vector_store)

    run_interface(transcribe, predict_icd)
