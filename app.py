import gradio as gr
import tempfile
import os
from ranker import rank_candidates

def run_ranking(file):
    if file is None:
        return None

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    ).name

    rank_candidates(
        file.name,
        output_file,
        top_n=100
    )

    return output_file

demo = gr.Interface(
    fn=run_ranking,
    inputs=gr.File(label="Upload candidates.jsonl"),
    outputs=gr.File(label="Download submission.csv"),
    title="RedRob Candidate Ranker",
    description="Upload candidates.jsonl and generate submission.csv"
)

if __name__ == "__main__":
    demo.launch()