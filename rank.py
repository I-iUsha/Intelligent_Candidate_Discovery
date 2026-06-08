"""
rank.py — entry point alias for ranker.py
Run: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
from ranker import rank_candidates
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of top candidates to output")
    args = parser.parse_args()

    start = time.time()
    rank_candidates(args.candidates, args.out, args.top)
    print(f"\nTotal runtime: {time.time() - start:.1f}s")
