"""Build the research notebook; execution is handled by 07_execute_notebook.py."""
from pathlib import Path
from study_content import sections
from study_document import build_notebook
def build():
    root=Path(__file__).resolve().parents[1]
    build_notebook(sections(root))
if __name__=='__main__':build()
