#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma

from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose import Pose
from pose_format.pose_header import (
    PoseHeader,
    PoseHeaderComponent,
    PoseHeaderDimensions,
)

from pose_format.utils.alphapose import get_alphapose_136_components
from pose_format.utils.cocowholebody133_header import cocowholebody_components
from pose_format.utils.openpose import OpenPose_Components
from pose_format.utils.sapiens import get_sapiens_components
from pose_format.utils.smplest_x import get_smplx_components


def _holistic_components():
    """MediaPipe holistic component definitions (hardcoded to avoid mediapipe dependency)."""
    body_points = [
        "NOSE",
        "LEFT_EYE_INNER",
        "LEFT_EYE",
        "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER",
        "RIGHT_EYE",
        "RIGHT_EYE_OUTER",
        "LEFT_EAR",
        "RIGHT_EAR",
        "MOUTH_LEFT",
        "MOUTH_RIGHT",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_PINKY",
        "RIGHT_PINKY",
        "LEFT_INDEX",
        "RIGHT_INDEX",
        "LEFT_THUMB",
        "RIGHT_THUMB",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    ]
    hand_points = [
        "WRIST",
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    ]
    face_points = [str(i) for i in range(468)]
    pf = "XYZC"
    return [
        PoseHeaderComponent(
            name="POSE_LANDMARKS",
            points=body_points,
            limbs=[],
            colors=[(255, 0, 0)],
            point_format=pf,
        ),
        PoseHeaderComponent(
            name="FACE_LANDMARKS",
            points=face_points,
            limbs=[],
            colors=[(128, 0, 0)],
            point_format=pf,
        ),
        PoseHeaderComponent(
            name="LEFT_HAND_LANDMARKS",
            points=hand_points,
            limbs=[],
            colors=[(0, 255, 0)],
            point_format=pf,
        ),
        PoseHeaderComponent(
            name="RIGHT_HAND_LANDMARKS",
            points=hand_points,
            limbs=[],
            colors=[(0, 0, 255)],
            point_format=pf,
        ),
        PoseHeaderComponent(
            name="POSE_WORLD_LANDMARKS",
            points=body_points,
            limbs=[],
            colors=[(255, 0, 0)],
            point_format=pf,
        ),
    ]


# mmposewholebody, openpifpaf, and sdpose all use the COCO WholeBody 133-point layout.
_POSE_TYPE_COMPONENTS = {
    "mediapipe": _holistic_components,
    "alphapose_136": get_alphapose_136_components,
    "mmposewholebody": cocowholebody_components,
    "openpifpaf": cocowholebody_components,
    "openpose": lambda: OpenPose_Components,
    "sapiens": get_sapiens_components,
    "sdpose": cocowholebody_components,
    "smplest_x": get_smplx_components,
}


def fake_pose(
    pose_type: str,
    num_frames: int = 30,
    num_people: int = 1,
    fps: float = 25.0,
) -> Pose:
    """Build a random Pose with the correct header structure for the given pose type."""
    components = _POSE_TYPE_COMPONENTS[pose_type]()
    dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    header = PoseHeader(version=0.2, dimensions=dimensions, components=components)

    total_points = header.total_points()
    num_dims = header.num_dims()
    data = np.random.randn(num_frames, num_people, total_points, num_dims).astype(
        np.float32
    )
    confidence = np.ones((num_frames, num_people, total_points), dtype=np.float32)

    body = NumPyPoseBody(fps=fps, data=ma.masked_array(data), confidence=confidence)
    return Pose(header, body)
