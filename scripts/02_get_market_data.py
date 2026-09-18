"""Run from any directory: python scripts/02_get_market_data.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.market import download

if __name__ == "__main__":
    download()
