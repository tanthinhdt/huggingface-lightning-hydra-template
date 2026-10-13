"""Aggregate explicit evaluation paths: python -m src.analyze a/metrics.json b/metrics.json."""

import argparse

from src.research.aggregation import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output", default="summary.json")
    args = parser.parse_args()
    print(aggregate(args.metrics, args.output))


if __name__ == "__main__":
    main()
