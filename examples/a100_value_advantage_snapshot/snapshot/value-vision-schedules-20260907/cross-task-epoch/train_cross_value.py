"""Reuse the unchanged lean trainer in a separate output namespace."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import train_lean_value as trainer
trainer.EXPERIMENT=ROOT
if __name__=='__main__':trainer.main()
