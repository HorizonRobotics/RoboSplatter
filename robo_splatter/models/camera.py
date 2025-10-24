# Project RoboSplatter
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation

__all__ = [
    "Camera",
]


@dataclass
class Camera:
    """A class for batch processing of camera parameters.

    Attributes:
        c2w: Camera-to-world transformation matrix, shape (batch, 4, 4).
        Ks: Camera intrinsic matrix, shape (batch, 3, 3).
        image_height: Height of the image in pixels.
        image_width: Width of the image in pixels.
        device: Device to store tensors (e.g., 'cpu', 'cuda').
    """

    c2w: torch.Tensor
    Ks: torch.Tensor
    image_height: int
    image_width: int
    device: str = "cuda"

    SIM_COORD_ALIGN = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


    def __post_init__(self) -> None:
        if isinstance(self.c2w, np.ndarray):
            self.c2w = torch.from_numpy(self.c2w).float()
        if isinstance(self.Ks, np.ndarray):
            self.Ks = torch.from_numpy(self.Ks).float()

        if self.c2w.ndim == 2:
            self.c2w = self.c2w.unsqueeze(0)
        if self.Ks.ndim == 2:
            self.Ks = self.Ks.unsqueeze(0)

        self.to(self.device)

    @classmethod
    def init_from_pose_list(
        cls,
        pose_list: np.ndarray,
        camera_intrinsic: np.ndarray,
        image_height: int,
        image_width: int,
        device: str = "cpu",
    ) -> "Camera":
        """Initialize from a list of poses and camera intrinsics.

        Args:
            pose_list: shape (batch, 7) with [x, y, z, qx, qy, qz, qw].
            camera_intrinsic: Array of shape (3, 3) or (batch, 3, 3).
            image_height: Image height in pixels.
            image_width: Image width in pixels.
            device: Device to store tensors.

        Returns:
            BatchCamera instance.
        """

        pose_list = np.asarray(pose_list)
        if pose_list.shape[-1] == 7:
            if pose_list.ndim == 1:
                pose_list = pose_list[np.newaxis]
            assert (
                pose_list.shape[-1] == 7
            ), f"Expected pose_list shape (*, 7), got {pose_list.shape}"
            batch_size = pose_list.shape[0]

            c2w = torch.eye(4, dtype=torch.float32).repeat(batch_size, 1, 1)
            rotations = Rotation.from_quat(
                pose_list[:, 3:]
            ).as_matrix()  # (batch, 3, 3)
            c2w[:, :3, :3] = torch.from_numpy(rotations).float()
            c2w[:, :3, 3] = torch.from_numpy(pose_list[:, :3]).float()
        elif pose_list.shape[-2:] == (4, 4):
            if pose_list.ndim == 2:
                pose_list = pose_list[np.newaxis]
            assert pose_list.shape[-2:] == (
                4,
                4,
            ), f"Expected pose_list shape (*, 4, 4), got {pose_list.shape}"
            batch_size = pose_list.shape[0]
            c2w = torch.from_numpy(pose_list).float()
        else:
            raise ValueError("pose_list must be shape [7] or [*,4,4]")

        camera_intrinsic = np.asarray(camera_intrinsic)
        if camera_intrinsic.ndim == 2:
            camera_intrinsic = np.repeat(
                camera_intrinsic[np.newaxis], batch_size, axis=0
            )
        assert camera_intrinsic.shape == (
            batch_size,
            3,
            3,
        ), f"Expected camera_intrinsic ({batch_size}, 3, 3), got {camera_intrinsic.shape}"  # noqa
        Ks = torch.from_numpy(camera_intrinsic).float()

        return cls(
            c2w=c2w,
            Ks=Ks,
            image_height=image_height,
            image_width=image_width,
            device=device,
        )

    @classmethod
    def init_from_pose_tensor(
        cls,
        c2w: torch.Tensor,
        Ks: torch.Tensor,
        image_height: int,
        image_width: int,
        device: Union[str, torch.device] = "cpu",
    ) -> "Camera":
        """Initialize directly from batched torch tensors for extrinsics and intrinsics.

        Args:
            c2w: Camera-to-world matrix tensor with shape (4, 4) or (B, 4, 4).
            Ks: Intrinsic matrix tensor with shape (3, 3) or (B, 3, 3).
            image_height: Image height in pixels.
            image_width: Image width in pixels.
            device: Target device for the tensors.

        Returns:
            Camera instance.
        """
        # Normalize device to string
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = device

        if c2w.ndim == 2:
            c2w = c2w.unsqueeze(0)
        if Ks.ndim == 2:
            Ks = Ks.unsqueeze(0).expand(c2w.shape[0], -1, -1)

        # Validate shapes
        assert c2w.shape[-2:] == (4, 4), f"Expected c2w (*,4,4), got {c2w.shape}"
        assert Ks.shape[-2:] == (3, 3), f"Expected Ks (*,3,3), got {Ks.shape}"
        assert c2w.shape[0] == Ks.shape[0], f"Batch size mismatch {c2w.shape[0]} vs {Ks.shape[0]}"

        c2w = c2w.to(device_str, dtype=torch.float32)
        Ks = Ks.to(device_str, dtype=torch.float32)

        return cls(
            c2w=c2w,
            Ks=Ks,
            image_height=image_height,
            image_width=image_width,
            device=device_str,
        )

    def to(self, device: Union[str, torch.device]) -> "Camera":
        # Normalize device
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = device

        if device_str.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device_str)
        self.device = device_str

        return self

    @property
    def sim_c2w(self) -> torch.Tensor:
        coord_align = (
            self.SIM_COORD_ALIGN.to(self.c2w)
            .unsqueeze(0)
            .repeat(self.c2w.shape[0], 1, 1)
        )
        return self.c2w @ coord_align
