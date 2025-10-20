# Audio-to-ICD *(A2ICD)*


**➡️ Checkout running model on HuggingFace 🤗: [NabJab/A2ICD](https://huggingface.co/spaces/NabJab/A2ICD).**

The app uses whisper-1 (OpenAI) and gpt-4.1 to transcribe audio to text. To predict ICD-10 codes gpt-4.1 is used with a database lookup.

**Input panels**
The Audio input panel allows users to record audio and transcribe it, while the Text input panel allows users to enter text directly.
![input panel](assets/panel-input.jpg "Input Panels")

**Output panel**
The Output panel displays the predicted ICD-10 codes and links them to the ICD website. The codes are ordered by importance.
![output panel](assets/panel-output.jpg "Output Panel")


## How to run
**1. Setup API key(s)**

Create a *.env* file in the root directory with the following content:
```OPENAI_API_KEY="your_openai_api_key"```

**2. Run app**
- Option 1: Using uv (recommended): `uv run mvp.py`
- Option 2: Using python
   - Install dependencies: `pip install -r requirements.txt`
   -  Run: `python mvp.py`

**3. Open in browser**
Go to http://localhost:7860
