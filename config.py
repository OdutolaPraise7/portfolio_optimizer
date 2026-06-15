from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PRICE_FILE = PROJECT_ROOT / "STOCK_PRICES.csv"      # ← change only this line
SIGNAL_FILE = PROJECT_ROOT / "signal_store.csv"   # ← and this one