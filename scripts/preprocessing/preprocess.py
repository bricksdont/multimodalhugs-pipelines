#!/usr/bin/env python3

import os
import argparse
import importlib
import logging
import csv
import itertools

from pose_format.utils.reader import BufferReader
from pose_format.pose import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody

from typing import Any, Iterator, Optional, Dict, List


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
    parser.add_argument("--fake", action="store_true", default=False,
                        help="Generate fake pose data instead of downloading a real dataset.", required=False)
    parser.add_argument("--num-fake-examples", type=int, default=5,
                        help="Number of fake examples to generate per split (only used with --fake).", required=False)
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
    import sign_language_datasets.datasets  # noqa: F401

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
    import tensorflow_datasets as tfds
    from sign_language_datasets.datasets.config import SignDatasetConfig

    config = SignDatasetConfig(name=dataset_name,
                               version="3.0.0",
                               include_video=False,
                               process_video=False,
                               fps=25,
                               include_pose=pose_type)

    dataset = tfds.load(dataset_name, builder_kwargs=dict(config=config), data_dir=data_dir)

    return dataset


Example = Dict[str, str]


def generate_examples(dataset: Any,
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


def _fake_holistic_pose(num_frames: int, num_people: int = 1, fps: float = 25.0) -> Pose:
    """
    Build a fake mediapipe holistic pose without importing mediapipe.
    Component names and point lists are the standard MediaPipe holistic definitions.
    """
    import numpy as np
    import numpy.ma as ma
    from pose_format.pose_header import PoseHeaderComponent, PoseHeaderDimensions

    body_points = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    ]
    hand_points = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]
    face_points = [str(i) for i in range(468)]

    pf = "XYZC"
    components = [
        PoseHeaderComponent(name="POSE_LANDMARKS", points=body_points, limbs=[], colors=[(255, 0, 0)], point_format=pf),
        PoseHeaderComponent(name="FACE_LANDMARKS", points=face_points, limbs=[], colors=[(128, 0, 0)], point_format=pf),
        PoseHeaderComponent(name="LEFT_HAND_LANDMARKS", points=hand_points, limbs=[], colors=[(0, 255, 0)], point_format=pf),
        PoseHeaderComponent(name="RIGHT_HAND_LANDMARKS", points=hand_points, limbs=[], colors=[(0, 0, 255)], point_format=pf),
        PoseHeaderComponent(name="POSE_WORLD_LANDMARKS", points=body_points, limbs=[], colors=[(255, 0, 0)], point_format=pf),
    ]

    dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    header = PoseHeader(version=0.2, dimensions=dimensions, components=components)

    total_points = header.total_points()
    num_dims = header.num_dims()
    data = np.random.randn(num_frames, num_people, total_points, num_dims).astype(np.float32)
    confidence = np.ones((num_frames, num_people, total_points), dtype=np.float32)

    body = NumPyPoseBody(fps=fps, data=ma.masked_array(data), confidence=confidence)
    return Pose(header, body)


def generate_fake_examples(feature_dir: str,
                            num_examples: int = 5,
                            num_frames: int = 30) -> List[Example]:
    """
    Generate fake holistic pose examples without downloading any real dataset.
    Does not require mediapipe. Useful for CI and smoke testing.

    :param feature_dir:
    :param num_examples:
    :param num_frames:
    :return:
    """
    examples = []

    for i in range(num_examples):
        datum_id = f"fake_{i:04d}"
        pose = _fake_holistic_pose(num_frames=num_frames)

        pose_filepath = os.path.join(feature_dir, f"{datum_id}.pose")

        with open(pose_filepath, "wb") as buffer:
            pose.write(buffer)

        examples.append({
            "datum_id": datum_id,
            "text": f"fake translation {i}",
            "pose_filepath": pose_filepath,
        })

    return examples


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

    stats = {}

    if args.fake:
        for split_name in ["train", "validation", "test"]:
            examples = generate_fake_examples(
                feature_dir=args.feature_dir,
                num_examples=args.num_fake_examples,
            )
            stats[split_name] = len(examples)
            write_examples_tsv(examples=examples,
                               output_dir=args.output_dir,
                               encoder_prompt=args.encoder_prompt,
                               decoder_prompt=args.decoder_prompt,
                               split_name=split_name)
    else:
        dataset = load_dataset(dataset_name=get_dataset_identifier(args.dataset),
                               data_dir=args.tfds_data_dir,
                               pose_type=get_pose_identifier(args.pose_type))

        pose_header = load_pose_header(dataset_name=get_dataset_identifier(args.dataset),
                                       pose_type=get_pose_identifier(args.pose_type))

        for split_name in ["train", "validation", "test"]:
            examples = list(generate_examples(dataset=dataset,
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
