#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import importlib
import logging
import csv
import itertools
import urllib.request

from pose_format.utils.reader import BufferReader
from pose_format.pose import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody

from typing import Any, Iterator, Optional, Dict, List

from fake_poses import fake_pose


SUPPORTED_DATASETS = ["phoenix"]
SUPPORTED_FEATURE_TYPES = ["pose"]
SUPPORTED_POSE_TYPES = [
    "alphapose_136",
    "mediapipe",
    "mmposewholebody",
    "openpifpaf",
    "openpose",
    "sapiens",
    "sdpose",
    "smplest_x",
]

ANNOTATIONS_URL = (
    "https://datasets.sigma-sign-language.com/public/phoenix/phoenix-annotations.tar.gz"
)

ANNOTATIONS_CSV_PATHS = {
    "train": "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv",
    "validation": "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv",
    "test": "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv",
}

POSE_DOWNLOAD_URLS = {
    "alphapose_136": "https://datasets.sigma-sign-language.com/poses/alphapose_136/phoenix.zip",
    "mediapipe": "https://datasets.sigma-sign-language.com/poses/holistic/phoenix.tar.gz",
    "mmposewholebody": "https://datasets.sigma-sign-language.com/poses/mmposewholebody/phoenix.zip",
    "openpifpaf": "https://datasets.sigma-sign-language.com/poses/openpifpaf/phoenix.zip",
    "openpose": "https://datasets.sigma-sign-language.com/poses/openpose/phoenix.zip",
    "sapiens": "https://datasets.sigma-sign-language.com/poses/sapiens/phoenix.zip",
    "sdpose": "https://datasets.sigma-sign-language.com/poses/sdpose/phoenix.zip",
    "smplest_x": "https://datasets.sigma-sign-language.com/poses/smplest_x/phoenix.zip",
}

# Expected feat_dim per pose type for sanity checking.
# mediapipe: 534 after reduce_holistic_poses transformation at runtime.
# Others: total_points * dims_per_keypoint as read from the body data array.
# Note: header format strings (e.g. XYC) may not match the actual stored dims;
# alphapose_136 declares XYC but only stores XY in the body.
# See https://github.com/ZurichNLP/video-to-pose for keypoint counts.
EXPECTED_FEAT_DIMS = {
    "mediapipe": 534,  # after reduce_holistic_poses
    "alphapose_136": 272,  # 136 * 2 (XY, 2D)
    "mmposewholebody": 266,  # 133 * 2 (XY)
    "openpifpaf": 266,  # 133 * 2 (XY)
    "sdpose": 266,  # 133 * 2 (XY)
    "openpose": 274,  # 137 * 2 (XY)
    "smplest_x": 278,  # 139 * 2 (XY)
    "sapiens": 620, # 310 * 2 (XY)
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and transform a TSV file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--feature-type", type=str, required=True, help="Feature type (e.g. 'pose')."
    )
    parser.add_argument(
        "--pose-type", type=str, required=True, help="Pose type (e.g. 'mediapipe')."
    )
    parser.add_argument("--feature-dir", type=str, help="Where to save features.")
    parser.add_argument("--output-dir", type=str, help="Path to the output TSV files.")
    parser.add_argument(
        "--encoder-prompt", type=str, default="__dgs__", help="encoder prompt string."
    )
    parser.add_argument(
        "--decoder-prompt", type=str, default="__de__", help="decoder prompt string."
    )

    parser.add_argument(
        "--tfds-data-dir",
        type=str,
        default=None,
        help="TFDS data folder to cache downloads.",
        required=False,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Process very few elements only.",
        required=False,
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        default=False,
        help="Generate fake pose data instead of downloading a real dataset.",
        required=False,
    )
    parser.add_argument(
        "--num-fake-examples",
        type=int,
        default=5,
        help="Number of fake examples to generate per split (only used with --fake).",
        required=False,
    )
    return parser.parse_args()


def get_pose_identifier(pose_type: str):
    if pose_type == "mediapipe":
        return "holistic"
    else:
        return pose_type


def get_dataset_identifier(dataset: str):
    if dataset == "phoenix":
        return "rwth_phoenix2014_t"
    else:
        return dataset


def load_pose_header(dataset_name: str, pose_type: str) -> PoseHeader:
    import sign_language_datasets.datasets  # noqa: F401

    dataset_module = importlib.import_module(
        f"sign_language_datasets.datasets.{dataset_name}.{dataset_name}"
    )

    if pose_type not in dataset_module._POSE_HEADERS:
        raise ValueError(
            f"Pose type not supported: '{pose_type}'. Supported: {dataset_module._POSE_HEADERS.keys()}"
        )

    with open(dataset_module._POSE_HEADERS[pose_type], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    return pose_header


def load_dataset(
    dataset_name: str = "rwth_phoenix2014_t",
    data_dir: Optional[str] = None,
    pose_type: str = "holistic",
):
    import tensorflow_datasets as tfds
    from sign_language_datasets.datasets.config import SignDatasetConfig

    config = SignDatasetConfig(
        name=dataset_name,
        version="3.0.0",
        include_video=False,
        process_video=False,
        fps=25,
        include_pose=pose_type,
    )

    dataset = tfds.load(
        dataset_name, builder_kwargs=dict(config=config), data_dir=data_dir
    )

    return dataset


def load_text_labels(annotations_dir: str) -> Dict[str, Dict[str, str]]:
    """Download Phoenix annotations and return {split: {id: text}}, no TF required."""
    import tarfile

    archive_name = os.path.basename(ANNOTATIONS_URL)
    archive_path = os.path.join(annotations_dir, archive_name)
    tmp_path = archive_path + ".tmp"

    if not os.path.exists(archive_path):
        logging.info(f"Downloading annotations from {ANNOTATIONS_URL}")
        request = urllib.request.Request(
            ANNOTATIONS_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as f:
            f.write(response.read())
        os.rename(tmp_path, archive_path)

    first_csv = os.path.join(
        annotations_dir, next(iter(ANNOTATIONS_CSV_PATHS.values()))
    )
    if not os.path.exists(first_csv):
        logging.info(f"Extracting {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(annotations_dir)

    labels: Dict[str, Dict[str, str]] = {}
    for split_name, csv_rel_path in ANNOTATIONS_CSV_PATHS.items():
        csv_path = os.path.join(annotations_dir, csv_rel_path)
        labels[split_name] = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                labels[split_name][row["name"]] = row["translation"]

    return labels


Example = Dict[str, str]


def generate_examples(
    dataset: Any,
    split_name: str,
    pose_header: PoseHeader,
    feature_dir: str,
    dry_run: bool = False,
) -> Iterator[Example]:
    if dry_run:
        data_iterator = itertools.islice(dataset[split_name], 0, 10)
    else:
        data_iterator = dataset[split_name]

    for datum in data_iterator:
        datum_id = datum["id"].numpy().decode("utf-8")

        text = datum["text"].numpy().decode("utf-8")

        pose_data = datum["pose"]["data"].numpy()
        pose_confidence = datum["pose"]["conf"].numpy()

        fps = int(datum["pose"]["fps"].numpy())

        pose_body = NumPyPoseBody(fps=fps, data=pose_data, confidence=pose_confidence)

        pose = Pose(pose_header, pose_body)

        pose_filepath = os.path.join(feature_dir, f"{datum_id}.pose")

        if dry_run:
            logging.debug(f"Writing pose to: '{pose_filepath}'")

        with open(pose_filepath, "wb") as data_buffer:
            pose.write(data_buffer)

        example = {"datum_id": datum_id, "text": text, "pose_filepath": pose_filepath}

        yield example


def generate_examples_from_directory(
    feature_dir: str, split_name: str, text_labels: Dict[str, str]
) -> List[Example]:
    """Enumerate pre-downloaded .pose files and match with text labels."""
    split_dir = os.path.join(feature_dir, split_name)
    pose_files = sorted(glob.glob(os.path.join(split_dir, "*.pose")))

    examples = []
    for pose_filepath in pose_files:
        datum_id = os.path.splitext(os.path.basename(pose_filepath))[0]
        text = text_labels.get(datum_id, "")
        if not text:
            logging.warning(f"No text label found for id: {datum_id}")
        examples.append(
            {"datum_id": datum_id, "text": text, "pose_filepath": pose_filepath}
        )

    return examples


class DownloadError(Exception):
    """Raised when a network download fails; signals the shell retry loop."""


def _promote_subdirectory(feature_dir: str) -> None:
    """If extraction produced a single top-level subdirectory, move its contents up."""
    if os.path.isdir(os.path.join(feature_dir, "train")):
        return
    subdirs = [
        d
        for d in os.listdir(feature_dir)
        if os.path.isdir(os.path.join(feature_dir, d))
    ]
    if len(subdirs) == 1:
        subdir_path = os.path.join(feature_dir, subdirs[0])
        for item in os.listdir(subdir_path):
            os.rename(os.path.join(subdir_path, item), os.path.join(feature_dir, item))
        os.rmdir(subdir_path)
        logging.info(f"Promoted contents of {subdirs[0]}/ up to {feature_dir}/")


def download_and_extract_poses(pose_type: str, feature_dir: str) -> None:
    """Download archive from Cloudflare and extract to feature_dir."""
    url = POSE_DOWNLOAD_URLS[pose_type]
    archive_name = os.path.basename(url)
    archive_path = os.path.join(feature_dir, archive_name)
    tmp_path = archive_path + ".tmp"

    if not os.path.exists(archive_path):
        logging.info(f"Downloading {url} -> {archive_path}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as f:
                f.write(response.read())
        except Exception as e:
            raise DownloadError(f"Failed to download {url}: {e}") from e
        os.rename(tmp_path, archive_path)
    else:
        logging.info(f"Archive already exists, skipping download: {archive_path}")

    train_dir = os.path.join(feature_dir, "train")
    if os.path.isdir(train_dir) and any(
        f.endswith(".pose") for f in os.listdir(train_dir)
    ):
        logging.info(f"Poses already extracted in {train_dir}, skipping extraction")
        return

    logging.info(f"Extracting {archive_path} -> {feature_dir}")
    if archive_name.endswith(".tar.gz"):
        import tarfile

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(feature_dir)
    else:
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(feature_dir)

    _promote_subdirectory(feature_dir)

    # mediapipe archive uses "dev/" instead of "validation/"
    dev_dir = os.path.join(feature_dir, "dev")
    val_dir = os.path.join(feature_dir, "validation")
    if os.path.isdir(dev_dir) and not os.path.isdir(val_dir):
        logging.info("Renaming dev/ -> validation/")
        os.rename(dev_dir, val_dir)


def compute_feat_dim_from_file(pose_filepath: str) -> int:
    """Compute feat_dim from body.data.shape and cross-check against the header.

    The header's point_format (e.g. "XYC") declares how many values are stored
    per point. A trailing "C" means confidence, which pose-format strips into
    body.confidence — so the expected coordinate dims are len(format) - 1 when
    the format ends in "C", otherwise len(format).  We compare this against the
    actual body.data.shape[-1] and warn on any mismatch.
    """
    with open(pose_filepath, "rb") as f:
        pose = Pose.read(f.read())
    _, _, n_points, n_dims = pose.body.data.shape

    for component in pose.header.components:
        fmt = component.format
        expected_dims = len(fmt) - 1 if fmt.endswith("C") else len(fmt)
        if expected_dims != n_dims:
            logging.warning(
                f"Component {component.name!r}: header format {fmt!r} implies "
                f"{expected_dims} coordinate dims, but body.data has {n_dims}. "
                f"Pose file may be malformed: {pose_filepath}"
            )

    return n_points * n_dims


def get_feat_dim(pose_type: str, feature_dir: str) -> int:
    """Return feat_dim, warn if it differs from the expected value."""
    if pose_type == "mediapipe":
        # reduce_holistic_poses is applied at runtime by the processor; result is always 534
        feat_dim = EXPECTED_FEAT_DIMS["mediapipe"]
    else:
        pose_files = sorted(glob.glob(os.path.join(feature_dir, "train", "*.pose")))
        if not pose_files:
            raise FileNotFoundError(f"No .pose files found in {feature_dir}/train")
        feat_dim = compute_feat_dim_from_file(pose_files[0])

    expected = EXPECTED_FEAT_DIMS.get(pose_type)
    if expected is not None and feat_dim != expected:
        logging.warning(
            f"feat_dim={feat_dim} for pose_type={pose_type!r} "
            f"does not match expected value {expected}. Please check the pose header."
        )
    return feat_dim


def generate_fake_examples(
    feature_dir: str, pose_type: str, num_examples: int = 5, num_frames: int = 30
) -> List[Example]:
    """
    Generate fake pose examples for the given pose_type without downloading any data.
    Useful for CI and smoke testing.
    """
    examples = []

    for i in range(num_examples):
        datum_id = f"fake_{i:04d}"
        pose = fake_pose(pose_type=pose_type, num_frames=num_frames)

        pose_filepath = os.path.join(feature_dir, f"{datum_id}.pose")

        with open(pose_filepath, "wb") as buffer:
            pose.write(buffer)

        examples.append(
            {
                "datum_id": datum_id,
                "text": f"fake translation {i}",
                "pose_filepath": pose_filepath,
            }
        )

    return examples


def write_examples_tsv(
    examples: List[Example],
    output_dir: str,
    encoder_prompt: str,
    decoder_prompt: str,
    split_name: str,
):
    """
    If signal_start and signal_end are not required (when all the frames are used), must be set as 0.
    """

    filepath = os.path.join(output_dir, f"{split_name}.tsv")

    logging.debug("Writing generated examples to: '%s'" % filepath)

    fieldnames = [
        "signal",
        "signal_start",
        "signal_end",
        "encoder_prompt",
        "decoder_prompt",
        "output",
    ]

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
                "output": example["text"],
            }

            writer.writerow(row_dict)


def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: '{args.dataset}'. Supported: {SUPPORTED_DATASETS}"
        )

    if args.feature_type not in SUPPORTED_FEATURE_TYPES:
        raise ValueError(
            f"Unsupported feature type: '{args.feature_type}'. Supported: {SUPPORTED_FEATURE_TYPES}"
        )

    if args.pose_type not in SUPPORTED_POSE_TYPES:
        raise ValueError(
            f"Unsupported pose type: '{args.pose_type}'. Supported: {SUPPORTED_POSE_TYPES}"
        )

    stats = {}

    if args.fake:
        for split_name in ["train", "validation", "test"]:
            examples = generate_fake_examples(
                feature_dir=args.feature_dir,
                pose_type=args.pose_type,
                num_examples=args.num_fake_examples,
            )
            stats[split_name] = len(examples)
            write_examples_tsv(
                examples=examples,
                output_dir=args.output_dir,
                encoder_prompt=args.encoder_prompt,
                decoder_prompt=args.decoder_prompt,
                split_name=split_name,
            )
    else:
        try:
            download_and_extract_poses(args.pose_type, args.feature_dir)
        except DownloadError as e:
            logging.error(str(e))
            sys.exit(2)

        # annotations_dir = $base/data/$dataset (two levels above feature_dir)
        annotations_dir = os.path.dirname(os.path.dirname(args.feature_dir))
        text_labels = load_text_labels(annotations_dir=annotations_dir)

        for split_name in ["train", "validation", "test"]:
            examples = generate_examples_from_directory(
                feature_dir=args.feature_dir,
                split_name=split_name,
                text_labels=text_labels[split_name],
            )
            if args.dry_run:
                examples = examples[:10]
            stats[split_name] = len(examples)
            write_examples_tsv(
                examples=examples,
                output_dir=args.output_dir,
                encoder_prompt=args.encoder_prompt,
                decoder_prompt=args.decoder_prompt,
                split_name=split_name,
            )

        feat_dim = get_feat_dim(args.pose_type, args.feature_dir)
        feat_dim_path = os.path.join(args.output_dir, "feat_dim.txt")
        with open(feat_dim_path, "w") as f:
            f.write(str(feat_dim))
        logging.info(f"feat_dim={feat_dim} written to {feat_dim_path}")

    logging.debug("Number of examples found:")
    logging.debug(stats)


if __name__ == "__main__":
    main()
