# Multilingual Fake News Detector — app.py
"""Gradio UI for the Multilingual Fake News Detector.

Loads the model and tokenizer exactly once via model_loader and exposes
a Gradio Blocks interface for English, Urdu, and Spanish text.
"""

import gradio as gr

from model_loader import (
    LABELS,
    SUPPORTED_LANGUAGES,
    load_model,
    load_tokenizer,
    predict,
)

load_model()
load_tokenizer()


def analyse(text: str, language: str):
    """Run a prediction for the Gradio interface.

    Args:
        text: The user-supplied text to classify.
        language: The language label selected in the dropdown.

    Returns:
        A tuple of (summary string, dict of per-class confidence scores)
        suitable for the Textbox and gr.Label outputs.
    """
    if text is None or not text.strip():
        return "Error: please enter some text to analyse.", {}

    try:
        result = predict(text)
    except ValueError as exc:
        return f"Error: {exc}", {}
    except RuntimeError as exc:
        return f"Inference error: {exc}", {}
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors to the UI
        return f"Unexpected error: {exc}", {}

    prediction = result["prediction"]
    confidence_pct = round(result["confidence"] * 100, 2)
    summary = (
        f"Prediction: {prediction} | Confidence: {confidence_pct}% "
        f"| Language: {language}"
    )
    return summary, result["scores"]


EXAMPLES = [
    [
        "BREAKING: Scientists confirm the moon is made of cheese after secret NASA mission.",
        "English",
    ],
    [
        "وزیر اعظم نے آج قوم سے خطاب میں نئی اقتصادی پالیسی کا اعلان کیا۔",
        "Urdu",
    ],
    [
        "El gobierno anuncia un nuevo plan económico para impulsar el empleo juvenil.",
        "Spanish",
    ],
]

DESCRIPTION = (
    "Detect whether a news headline or short article is **Real** or **Fake** "
    "across three languages: English, Urdu, and Spanish.\n\n"
    "Built on a fine-tuned `bert-base-multilingual-cased` model with **90% "
    "overall accuracy** on the held-out test set."
)

with gr.Blocks(title="Multilingual Fake News Detector") as demo:
    gr.Markdown("# Multilingual Fake News Detector")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                lines=4,
                label="News text",
                placeholder="Paste a headline or short article in English, Urdu, or Spanish...",
            )
            language_dropdown = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="English",
                label="Language",
            )
            analyse_btn = gr.Button("Analyse", variant="primary")

        with gr.Column():
            summary_output = gr.Textbox(label="Result", interactive=False)
            scores_output = gr.Label(label="Confidence scores", num_top_classes=len(LABELS))

    gr.Examples(examples=EXAMPLES, inputs=[text_input, language_dropdown])

    analyse_btn.click(
        fn=analyse,
        inputs=[text_input, language_dropdown],
        outputs=[summary_output, scores_output],
    )


if __name__ == "__main__":
    demo.launch()
