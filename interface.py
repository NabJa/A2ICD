import gradio as gr


def run_interface(transcribe, predict_icd):
    """Run the Gradio interface for audio transcription and ICD-10 code prediction."""

    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            # Audio input and transcription
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload or record audio")
                transcribe_btn = gr.Button("Transcribe")

            # ICD-10 code prediction
            with gr.Column():
                transcribed_text = gr.Textbox(
                    label="Transcribed", interactive=True
                )  # Now editable!
                icd_btn = gr.Button("Get ICD-10 Codes")

            # Display ICD-10 codes
            with gr.Column():
                icd10_codes = gr.Markdown(label="ICD-10 Code(s)")

        # CALLBACK FUNCTIONS
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
