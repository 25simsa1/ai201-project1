"""
Milestone 5 - Gradio web interface.

Run with:  python app.py
Then open:  http://localhost:7860
"""

import gradio as gr

from query import ask


def handle_query(question):
    if not question or not question.strip():
        return "Please type a question first.", ""

    result = ask(question)

    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(no sources - the reviews didn't cover this)"

    return result["answer"], sources


with gr.Blocks(title="Colby Professor Reviews Q&A") as demo:
    gr.Markdown(
        "# Colby Professor Reviews Q&A\n"
        "Ask about Colby professors and get answers grounded in student reviews "
        "from Rate My Professors. Answers come only from the retrieved reviews."
    )

    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. Which professors are engaging in lectures?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(
        examples=[
            "Which Colby professors are known for being engaging in lectures?",
            "What do students say about Professor Findlay's grading?",
            "Which professors have the lightest workload?",
        ],
        inputs=inp,
    )

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
