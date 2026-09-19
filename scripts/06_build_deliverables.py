"""Build the standalone report and notebook from the paper-replication outputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.study_figures import build as figures
from study_document import build
if __name__=='__main__':
    figures()
    build()
