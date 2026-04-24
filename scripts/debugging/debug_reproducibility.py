import argparse
import torch  # noqa: F401
import multimodalhugs.models  # noqa: F401

from transformers import AutoModelForSeq2SeqLM


def get_diff(sd1, sd2, k):
    """
    If the two tensors are bit-identical, the result will be:

    0.0


    If they differ slightly because of floating point noise you might see e.g.:

    1e-7 or 1e-8


    If they differ because of different initialization / different update order / different kernels, you might see:

    1e-3 or 1e-1 or >1

    :param sd1:
    :param sd2:
    :return:
    """
    diff = (sd1[k] - sd2[k]).abs().max()
    print("MAGNITUDE: ", k, diff.item())


def check_checkpoints_zero_identical(checkpoint_1, checkpoint_2):

    m1 = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_1)
    m2 = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_2)

    sd1 = m1.state_dict()
    sd2 = m2.state_dict()

    all_equal = True

    for k in sd1.keys():
        if not (sd1[k].equal(sd2[k])):
            print("MISMATCH:", k)
            get_diff(sd1, sd2, k)
            all_equal = False

    if all_equal:
        print("✅ all parameters are identical")
    else:
        print("❌ at least one parameter differed")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Debug HF models")
    parser.add_argument("--checkpoint-1", type=str, help="Path to model checkpoint")
    parser.add_argument("--checkpoint-2", type=str, help="Path to model checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    check_checkpoints_zero_identical(args.checkpoint_1, args.checkpoint_2)
