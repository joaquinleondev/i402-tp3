import random
from copy import deepcopy

import numpy as np


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class particle():

    def __init__(self):
        self.x = (random.random() - 0.5) * 2
        self.y = (random.random() - 0.5) * 2
        self.orientation = random.uniform(-np.pi, np.pi)
        self.weight = 1.0

    def set(self, new_x, new_y, new_orientation):
        """Set particle pose."""
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def move_odom(self, odom, noise):
        """Move particle using the noisy odometry model."""
        dist = odom['t']
        delta_rot1 = odom['r1']
        delta_rot2 = odom['r2']

        alpha1, alpha2, alpha3, alpha4 = noise

        rot1_var = alpha1 * delta_rot1 ** 2 + alpha2 * dist ** 2
        trans_var = alpha3 * dist ** 2 + alpha4 * (
            delta_rot1 ** 2 + delta_rot2 ** 2
        )
        rot2_var = alpha1 * delta_rot2 ** 2 + alpha2 * dist ** 2

        noisy_rot1 = delta_rot1 + np.random.normal(
            0.0,
            np.sqrt(max(rot1_var, 0.0))
        )
        noisy_dist = dist + np.random.normal(
            0.0,
            np.sqrt(max(trans_var, 0.0))
        )
        noisy_rot2 = delta_rot2 + np.random.normal(
            0.0,
            np.sqrt(max(rot2_var, 0.0))
        )

        x_new = self.x + noisy_dist * np.cos(self.orientation + noisy_rot1)
        y_new = self.y + noisy_dist * np.sin(self.orientation + noisy_rot1)
        theta_new = normalize_angle(
            self.orientation + noisy_rot1 + noisy_rot2
        )

        self.set(x_new, y_new, theta_new)

    def set_weight(self, weight):
        """Set particle importance weight."""
        self.weight = float(weight)


class RobotFunctions:

    def __init__(self, num_particles=0):
        self.num_particles = int(num_particles)
        self.particles = []

        for _ in range(self.num_particles):
            self.particles.append(particle())

    def get_weights(self):
        if self.num_particles == 0:
            return np.array([])

        weights = np.array(
            [p.weight for p in self.particles],
            dtype=np.float64
        )
        total = np.sum(weights)

        if not np.isfinite(total) or total <= 0.0:
            return np.ones(self.num_particles) / self.num_particles

        return weights / total

    def get_particle_states(self):
        if self.num_particles == 0:
            return np.array([])

        return np.array(
            [[p.x, p.y, p.orientation] for p in self.particles]
        )

    def move_particles(self, deltas):
        for part in self.particles:
            part.move_odom(deltas, [0.2, 0.2, 0.002, 0.002])

    def get_selected_state(self):
        """Estimate robot pose from the weighted particle set."""
        if self.num_particles == 0:
            return [0.0, 0.0, 0.0]

        weights = self.get_weights()
        states = self.get_particle_states()

        mean_x = np.sum(weights * states[:, 0])
        mean_y = np.sum(weights * states[:, 1])
        mean_theta = np.arctan2(
            np.sum(weights * np.sin(states[:, 2])),
            np.sum(weights * np.cos(states[:, 2]))
        )

        return [float(mean_x), float(mean_y), float(mean_theta)]

    def update_particles(self, data, map_data, grid):
        """Update particle weights from a likelihood field and resample."""
        if self.num_particles == 0:
            return

        ranges, angles = self._valid_scan_vectors(
            data.ranges,
            data.range_min,
            data.range_max,
            data.angle_min,
            data.angle_increment,
            max_beams=80
        )

        if len(ranges) == 0:
            return

        resolution = map_data.info.resolution
        origin_x = map_data.info.origin.position.x
        origin_y = map_data.info.origin.position.y
        height, width = grid.shape

        # The random term keeps one bad beam from eliminating a particle.
        z_hit = 0.95
        z_rand = 0.05
        log_weights = np.empty(self.num_particles, dtype=np.float64)

        for index, part in enumerate(self.particles):
            beam_angles = part.orientation + angles
            end_x = part.x + ranges * np.cos(beam_angles)
            end_y = part.y + ranges * np.sin(beam_angles)

            cols = np.floor((end_x - origin_x) / resolution).astype(np.int64)
            rows = np.floor((end_y - origin_y) / resolution).astype(np.int64)

            in_bounds = (
                (rows >= 0) & (rows < height) &
                (cols >= 0) & (cols < width)
            )

            likelihoods = np.zeros(len(ranges), dtype=np.float64)
            likelihoods[in_bounds] = (
                grid[rows[in_bounds], cols[in_bounds]] / 100.0
            )

            beam_probs = np.clip(z_hit * likelihoods + z_rand, 1e-12, 1.0)
            log_weights[index] = np.sum(np.log(beam_probs))

        weights = self._normalize_log_weights(log_weights)

        for part, weight in zip(self.particles, weights):
            part.set_weight(weight)

        self.particles = self._systematic_resample(weights)

    def scan_refererence(
        self,
        ranges,
        range_min,
        range_max,
        angle_min,
        angle_max,
        angle_increment,
        last_odom
    ):
        """Transform LaserScan endpoints to map coordinates."""
        tx, ty, theta = last_odom
        valid_ranges, valid_angles = self._valid_scan_vectors(
            ranges,
            range_min,
            range_max,
            angle_min,
            angle_increment,
            max_beams=None
        )

        x_map = tx + valid_ranges * np.cos(theta + valid_angles)
        y_map = ty + valid_ranges * np.sin(theta + valid_angles)

        return np.array([x_map, y_map])

    def _valid_scan_vectors(
        self,
        ranges,
        range_min,
        range_max,
        angle_min,
        angle_increment,
        max_beams=None
    ):
        ranges = np.asarray(ranges, dtype=np.float64)
        angles = (
            angle_min +
            np.arange(len(ranges), dtype=np.float64) * angle_increment
        )

        # my_localization shifts the scan by pi before calling this class.
        angles = angles - np.pi

        valid = (
            np.isfinite(ranges) &
            (ranges >= range_min) &
            (ranges <= range_max)
        )
        indices = np.flatnonzero(valid)

        if max_beams is not None and len(indices) > max_beams:
            selected = np.linspace(
                0,
                len(indices) - 1,
                max_beams
            ).astype(np.int64)
            indices = indices[selected]

        return ranges[indices], angles[indices]

    def _normalize_log_weights(self, log_weights):
        max_log_weight = np.max(log_weights)

        if not np.isfinite(max_log_weight):
            return np.ones(self.num_particles) / self.num_particles

        weights = np.exp(log_weights - max_log_weight)
        total = np.sum(weights)

        if not np.isfinite(total) or total <= 0.0:
            return np.ones(self.num_particles) / self.num_particles

        return weights / total

    def _systematic_resample(self, weights):
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0

        step = 1.0 / self.num_particles
        start = np.random.uniform(0.0, step)
        positions = start + step * np.arange(self.num_particles)
        indices = np.searchsorted(cumulative, positions, side='left')

        new_particles = []
        uniform_weight = 1.0 / self.num_particles

        for index in indices:
            new_particle = deepcopy(self.particles[int(index)])
            new_particle.set_weight(uniform_weight)
            new_particles.append(new_particle)

        return new_particles
