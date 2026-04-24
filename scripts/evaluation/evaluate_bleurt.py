from evaluate import load
import numpy as np
import logging


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--references", type=str, required=True)
    parser.add_argument(
        "--checkpoint", type=str, required=False, default="bleurt-tiny-128"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    bleurt = load("bleurt", module_type="metric", config_name=args.checkpoint)

    predictions = open(args.predictions, "r").readlines()
    references = open(args.references, "r").readlines()

    results = bleurt.compute(predictions=predictions, references=references)

    average_score = np.mean(results["scores"])

    print(average_score)


if __name__ == "__main__":
    main()
