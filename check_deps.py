import sys
try:
    import evaluate
    import bert_score
    print("Dependencies installed")
except ImportError as e:
    print(f"Missing: {e}")
