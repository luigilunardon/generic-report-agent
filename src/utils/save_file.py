"Save file utility functions."

import json
from dataclasses import asdict
from pathlib import Path

import pypandoc

from constants import OUTPUT_DIR


def mk_output_dir(name):
    """Generate the output directory.

    Args:
        name (str): Name of the output directory.

    """
    directory = OUTPUT_DIR / name.replace(" ", "_")
    Path.mkdir(directory, exist_ok=True, parents=True)
    return directory


def save_md(content, directory, file_name="report.md"):
    """Save individual summary texts into separate files for each topic.

    Args:
        content (str): Content to save.
        directory (Path): Path of the directory.
        file_name (str): Name of the file.

    """
    path = directory / file_name
    with Path.open(path, "w", encoding='utf-8') as file:
        file.write(content)


def md_to_docx(directory, file_name="report.md"):
    """Convert .md files to .docx files following a template.

    Args:
        directory (Path): Path of the output directory.
        file_name (str): The name of the file to convert.

    """
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        pypandoc.download_pandoc()

    path_md = directory / file_name
    path_docx = directory / file_name.replace(".md", ".docx")

    with Path.open(path_md, encoding='utf-8') as f:
        text_md = f.read()

    extra_args = [f"--reference-doc={OUTPUT_DIR / 'template.docx'}"]

    pypandoc.convert_text(
        text_md, to='docx', format='md', outputfile=path_docx, extra_args=extra_args
    )


def save_state(state, path):
    """Save state of the search in a json file.

    Args:
        state (OverallState): State of the graph.
        path (str): Path of the recovery file.

    """
    with Path.open(Path(path), 'w', encoding='utf-8') as f:
        json.dump(asdict(state), f, indent=2)
