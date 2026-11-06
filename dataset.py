import numpy as np
import SimpleITK as sitk
from torch.utils import data

import utils


class TrainData(data.Dataset):
    """Sample sinograms and corresponding detector ray coordinates."""

    def __init__(
        self,
        theta,
        sin_path,
        sample_N,
        radius,
        field_of_view,
        speed_of_sound,
        time_point,
        num_rays,
        delay_sample,
        delta_t,
    ):
        """Initialize the dataset.

        Args:
            theta: Number of detector angles.
            sin_path: Path to the NIfTI sinogram.
            sample_N: Number of samples drawn from each sinogram row.
            radius: Detector array radius.
            field_of_view: Width of the detector field of view.
            speed_of_sound: Speed of sound used for time-distance conversion.
            time_point: Sampling interval in seconds.
            num_rays: Number of rays across the detector arc.
            delay_sample: Delay in sample units.
            delta_t: Small time offset used for forward differencing.
        """
        self.sample_N = int(sample_N)
        self.radius = radius
        self.speed_of_sound = speed_of_sound
        self.time_point = time_point
        self.num_rays = num_rays
        self.delay_sample = delay_sample
        self.delta_t = delta_t

        # Uniform detector positions over [0, 2*pi).
        self.sensor_angles = np.linspace(0, 360, int(theta), endpoint=False)

        # The sinogram has shape (num_angles, num_samples).
        sinogram = sitk.GetArrayFromImage(sitk.ReadImage(sin_path))
        self.num_angles, self.L = sinogram.shape
        if self.sensor_angles.size != self.num_angles:
            raise ValueError(
                f"theta={theta} does not match the {self.num_angles} "
                "sinogram angles."
            )
        if self.sample_N > self.L:
            raise ValueError(
                f"sample_N={sample_N} exceeds the {self.L} sinogram samples."
            )
        self.projections_lines = sinogram

        # Maximum detector-array angle subtended by half the field of view.
        angle_max = np.arcsin(field_of_view / (2 * radius))
        self.ray_angles = np.linspace(-angle_max, angle_max, num_rays)
        d_angle = angle_max / num_rays

        # Convert time samples to distances and integration weights.
        self.len = (
            (np.arange(self.L) + delay_sample)
            * time_point
            * speed_of_sound
        )
        self.len_plus = self.len + delta_t * speed_of_sound
        self.len_minus = self.len - delta_t * speed_of_sound
        self.drs_plus = 2 * self.len_plus * d_angle
        self.drs_minus = 2 * self.len_minus * d_angle

        # Precompute rays at zero degrees and rotate them for each projection.
        print("Building base ray coordinate template...")
        self.base_ray_plus = utils.build_coordinate_train(
            L=self.L,
            angle=0,
            radius=self.radius,
            time_point=self.time_point,
            ray_angles=self.ray_angles,
            speed_of_sound=self.speed_of_sound,
            delay_sample=self.delay_sample,
            dt=self.delta_t,
            num_rays=self.num_rays,
        )
        self.base_ray_minus = utils.build_coordinate_train(
            L=self.L,
            angle=0,
            radius=self.radius,
            time_point=self.time_point,
            ray_angles=self.ray_angles,
            speed_of_sound=self.speed_of_sound,
            delay_sample=self.delay_sample,
            dt=-self.delta_t,
            num_rays=self.num_rays,
        )

    def __len__(self):
        return self.num_angles

    def __getitem__(self, index):
        angle_rad = np.deg2rad(self.sensor_angles[index])
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        def rotate(base_coords):
            x = base_coords[..., 0]
            y = base_coords[..., 1]
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            return np.stack([new_x, new_y], axis=-1)

        ray_plus = rotate(self.base_ray_plus)
        ray_minus = rotate(self.base_ray_minus)

        sample_indices = np.random.choice(self.L, self.sample_N, replace=False)
        projection_l = self.projections_lines[index]

        return (
            projection_l[sample_indices].astype(np.float32),
            ray_plus[sample_indices].astype(np.float32),
            ray_minus[sample_indices].astype(np.float32),
            self.drs_plus[sample_indices].astype(np.float32),
            self.drs_minus[sample_indices].astype(np.float32),
            self.len_plus[sample_indices].astype(np.float32),
            self.len_minus[sample_indices].astype(np.float32),
        )
