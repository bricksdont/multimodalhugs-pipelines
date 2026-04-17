#!/usr/bin/env python3

import os
import argparse
import importlib
import logging
import csv
import itertools

import tensorflow as tf
import tensorflow_datasets as tfds
import sign_language_datasets.datasets

from pose_format.utils.reader import BufferReader
from pose_format.pose import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from sign_language_datasets.datasets.config import SignDatasetConfig

from typing import Iterator, Optional, Dict, List


# Parse command-line arguments
SUPPORTED_DATASETS = ["phoenix"]
SUPPORTED_FEATURE_TYPES = ["pose"]
SUPPORTED_POSE_TYPES = ["mediapipe"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and transform a TSV file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--feature-type", type=str, required=True, help="Feature type (e.g. 'pose').")
    parser.add_argument("--pose-type", type=str, required=True, help="Pose type (e.g. 'mediapipe').")
    parser.add_argument("--feature-dir", type=str, help="Where to save features.")
    parser.add_argument("--output-dir", type=str, help="Path to the output TSV files.")
    parser.add_argument("--encoder-prompt", type=str, default="__dgs__", help="encoder prompt string.")
    parser.add_argument("--decoder-prompt", type=str, default="__de__", help="decoder prompt string.")

    parser.add_argument("--tfds-data-dir", type=str, default=None,
                        help="TFDS data folder to cache downloads.", required=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Process very few elements only.", required=False)
    return parser.parse_args()


def get_pose_identifier(pose_type: str):
    """

    :param pose_type: a simple name for a pose type such as "mediapipe"
    :return: an identifier for the pose type such as "holistic" that the library sign-language-datasets uses
    """
    if pose_type == "mediapipe":
        return "holistic"
    else:
        return pose_type


def get_dataset_identifier(dataset: str):
    """

    :param dataset: a simple name for a dataset such as "phoenix"
    :return: an identifier for the dataset such as "rwth_phoenix2014_t" that the library sign-language-datasets uses
    """
    if dataset == "phoenix":
        return "rwth_phoenix2014_t"
    else:
        return dataset


def load_pose_header(dataset_name: str,
                     pose_type: str) -> PoseHeader:
    """
    Workaround from:
    https://github.com/sign-language-processing/datasets/issues/84

    :param dataset_name: A dataset name from sign_language_datasets.datasets
    :param pose_type: For instance holistic or openpose

    :return:
    """
    # Dynamically import the dataset module
    dataset_module = importlib.import_module(f"sign_language_datasets.datasets.{dataset_name}.{dataset_name}")

    if pose_type not in dataset_module._POSE_HEADERS:
        raise ValueError(f"Pose type not supported: '{pose_type}'. Supported: {dataset_module._POSE_HEADERS.keys()}")

    # Read the pose header from the dataset's predefined file
    with open(dataset_module._POSE_HEADERS[pose_type], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    return pose_header


def load_dataset(dataset_name: str = "rwth_phoenix2014_t",
                 data_dir: Optional[str] = None,
                 pose_type: str = "holistic"):
    """
    :param dataset_name:
    :param data_dir:
    :param pose_type:

    :return:
    """

    config = SignDatasetConfig(name=dataset_name,
                               version="3.0.0",
                               include_video=False,
                               process_video=False,
                               fps=25,
                               include_pose=pose_type)

    rwth_phoenix2014_t = tfds.load(dataset_name, builder_kwargs=dict(config=config), data_dir=data_dir)

    return rwth_phoenix2014_t


Example = Dict[str, str]


def generate_examples(dataset: tf.data.Dataset,
                      split_name: str,
                      pose_header: PoseHeader,
                      feature_dir: str,
                      dry_run: bool = False) -> Iterator[Example]:
    """
    :param dataset:
    :param split_name: "train", "validation" or "test"
    :param pose_header:
    :param feature_dir:
    :param dry_run:
    :return:
    """

    if dry_run:
        data_iterator = itertools.islice(dataset[split_name], 0, 10)
    else:
        data_iterator = dataset[split_name]

    for datum in data_iterator:

        datum_id = datum["id"].numpy().decode('utf-8')

        text = datum['text'].numpy().decode('utf-8')

        pose_data = datum['pose']['data'].numpy()
        pose_confidence = datum['pose']['conf'].numpy()

        fps = int(datum['pose']['fps'].numpy())

        pose_body = NumPyPoseBody(fps=fps,
                                  data=pose_data,
                                  confidence=pose_confidence)

        # Construct Pose object and write to file
        pose = Pose(pose_header, pose_body)

        pose_filepath = os.path.join(feature_dir, f"{datum_id}.pose")

        if dry_run:
            logging.debug(f"Writing pose to: '{pose_filepath}'")

        with open(pose_filepath, "wb") as data_buffer:
            pose.write(data_buffer)

        example = {
            "datum_id": datum_id,
            "text": text,
            "pose_filepath": pose_filepath
        }

        yield example


def write_examples_tsv(examples: List[Example],
                       output_dir: str,
                       encoder_prompt: str,
                       decoder_prompt: str,
                       split_name: str,):
    """
    If signal_start and signal_end are not required (when all the frames are used), must be set as 0.

    :param examples:
    :param output_dir:
    :param encoder_prompt:
    :param decoder_prompt:
    :param split_name:
    :return:
    """

    filepath = os.path.join(output_dir, f"{split_name}.tsv")

    logging.debug("Writing generated examples to: '%s'" % filepath)

    fieldnames = ["signal", "signal_start", "signal_end", "encoder_prompt", "decoder_prompt", "output"]

    with open(filepath, "w", newline="") as outhandle:
        writer = csv.DictWriter(outhandle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for example in examples:
            row_dict = {
                "signal": example["pose_filepath"],
                "signal_start": 0,
                "signal_end": 0,
                "encoder_prompt": encoder_prompt,
                "decoder_prompt": decoder_prompt,
                "output": example["text"]
            }

            writer.writerow(row_dict)


def main():
    # Parse arguments
    args = parse_arguments()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: '{args.dataset}'. Supported: {SUPPORTED_DATASETS}")

    if args.feature_type not in SUPPORTED_FEATURE_TYPES:
        raise ValueError(f"Unsupported feature type: '{args.feature_type}'. Supported: {SUPPORTED_FEATURE_TYPES}")

    if args.pose_type not in SUPPORTED_POSE_TYPES:
        raise ValueError(f"Unsupported pose type: '{args.pose_type}'. Supported: {SUPPORTED_POSE_TYPES}")

    phoenix_with_poses = load_dataset(dataset_name=get_dataset_identifier(args.dataset),
                                      data_dir=args.tfds_data_dir,
                                      pose_type=get_pose_identifier(args.pose_type))

    pose_header = load_pose_header(dataset_name=get_dataset_identifier(args.dataset),
                                   pose_type=get_pose_identifier(args.pose_type))

    stats = {}

    for split_name in ["train", "validation", "test"]:
        examples = list(generate_examples(dataset=phoenix_with_poses,
                                          split_name=split_name,
                                          pose_header=pose_header,
                                          feature_dir=args.feature_dir,
                                          dry_run=args.dry_run))

        stats[split_name] = len(examples)

        write_examples_tsv(examples=examples,
                           output_dir=args.output_dir,
                           encoder_prompt=args.encoder_prompt,
                           decoder_prompt=args.decoder_prompt,
                           split_name=split_name)

    logging.debug("Number of examples found:")
    logging.debug(stats)


if __name__ == "__main__":
    main()