"""
Industrial Defect Dataset - Synthetic Benchmark Generator
==========================================================
Generates realistic high-resolution (512x512) industrial metal surface images
for fault detection benchmarking.

Simulates two primary industrial surface archetypes:
1. Brushed / Rolled Sheet Metal (directional striations, grain texture)
2. Sand-Cast / Milled Iron (granular roughness, isotropic speckle)

Simulates four key industrial defect categories:
1. Scratches (curvilinear grooves with specular ridge displacement)
2. Cracks (jagged, branching dendritic fractures)
3. Patches / Scale / Inclusions (oxidized irregular surface blotches)
4. Pitting / Gas Porosity (3D craters with shadow-highlight depth)
"""

import os
import random
from typing import Tuple, List, Optional
import cv2
import numpy as np
from tqdm import tqdm


class IndustrialSurfaceGenerator:
    """
    Synthesizes realistic raw industrial surface inspection images.
    """

    def __init__(self, image_size: Tuple[int, int] = (512, 512), seed: int = 42):
        self.image_size = image_size
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def _generate_lighting_gradient(self) -> np.ndarray:
        """Simulates non-uniform industrial illumination (ring-light / spot-light roll-off)."""
        h, w = self.image_size
        center_x = w * (0.4 + 0.2 * np.random.rand())
        center_y = h * (0.4 + 0.2 * np.random.rand())
        max_dist = np.sqrt(w**2 + h**2) / 1.5

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Smooth quadratic illumination drop
        vignette = 1.0 - 0.25 * (dist / max_dist) ** 1.5
        vignette = np.clip(vignette, 0.7, 1.1)

        # Subtle directional gradient across inspection conveyor
        angle = np.random.uniform(0, 2 * np.pi)
        grad = np.cos(angle) * (x - w / 2) / w + np.sin(angle) * (y - h / 2) / h
        lighting = vignette + 0.1 * grad
        return lighting

    def _generate_brushed_surface(self) -> np.ndarray:
        """Simulates cold-rolled / brushed sheet metal with directional grain."""
        h, w = self.image_size
        base_intensity = np.random.uniform(130, 175)

        # 1. High-frequency base metal noise
        noise = np.random.normal(0, 12, (h, w))

        # 2. Elongated directional grain
        grain_length = np.random.randint(35, 65)
        grain_kernel = np.zeros((grain_length, 3))
        grain_kernel[:, 1] = 1.0 / grain_length

        # Rotate grain by random slight tilt angle (e.g. -15 to +15 deg)
        tilt = np.random.uniform(-10, 10)
        rot_mat = cv2.getRotationMatrix2D((1, grain_length // 2), tilt, 1.0)
        rotated_kernel = cv2.warpAffine(grain_kernel, rot_mat, (grain_length, grain_length))
        k_sum = np.sum(rotated_kernel)
        if k_sum > 0:
            rotated_kernel /= k_sum

        # Filter noise to create continuous parallel brushed lines
        brushed = cv2.filter2D(noise, -1, rotated_kernel)

        # 3. Multi-scale rolled bands
        band_noise = np.random.normal(0, 1, (h, 1))
        band_noise = cv2.GaussianBlur(band_noise, (1, 31), 0)
        bands = np.tile(band_noise, (1, w)) * np.random.uniform(8, 15)

        # 4. Assemble composite surface
        surface = base_intensity + brushed * 2.2 + bands
        lighting = self._generate_lighting_gradient()
        surface = surface * lighting
        return np.clip(surface, 0, 255).astype(np.uint8)

    def _generate_cast_surface(self) -> np.ndarray:
        """Simulates sand-cast metal with isotropic grain roughness and micro-porosity."""
        h, w = self.image_size
        base_intensity = np.random.uniform(115, 160)

        # Fine granular noise
        fine_noise = np.random.normal(0, 14, (h, w))
        fine_noise = cv2.GaussianBlur(fine_noise, (3, 3), 0)

        # Medium granular mold pattern (sand grain impression)
        med_noise = np.random.normal(0, 18, (h // 2, w // 2))
        med_noise = cv2.resize(med_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        med_noise = cv2.GaussianBlur(med_noise, (5, 5), 0)

        # Coarse surface undulation
        coarse_noise = np.random.normal(0, 12, (h // 8, w // 8))
        coarse_noise = cv2.resize(coarse_noise, (w, h), interpolation=cv2.INTER_CUBIC)

        surface = base_intensity + fine_noise + 0.7 * med_noise + coarse_noise
        lighting = self._generate_lighting_gradient()
        surface = surface * lighting
        return np.clip(surface, 0, 255).astype(np.uint8)

    def generate_normal_surface(self) -> np.ndarray:
        """Generates a normal, non-defective industrial surface image."""
        if random.random() < 0.55:
            return self._generate_brushed_surface()
        else:
            return self._generate_cast_surface()

    # --- Defect Synthesis Algorithms ---

    def _inject_scratch(self, image: np.ndarray) -> np.ndarray:
        """
        Injects realistic scratch/abrasion:
        A continuous grooved trajectory with dark core and adjacent highlight rim.
        """
        h, w = image.shape[:2]
        canvas = image.astype(np.float32)

        num_scratches = random.randint(1, 3)
        for _ in range(num_scratches):
            # Start position and random trajectory
            x = np.random.uniform(0.1 * w, 0.9 * w)
            y = np.random.uniform(0.1 * h, 0.9 * h)
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.uniform(80, 280)
            steps = int(length)

            points = []
            cur_x, cur_y = x, y
            cur_angle = angle

            for _ in range(steps):
                cur_x += np.cos(cur_angle)
                cur_y += np.sin(cur_angle)
                cur_angle += np.random.uniform(-0.06, 0.06)  # slight curve
                if 0 <= cur_x < w and 0 <= cur_y < h:
                    points.append((int(cur_x), int(cur_y)))

            if len(points) < 5:
                continue

            pts = np.array(points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))

            thickness = random.randint(2, 4)
            depth_dark = random.uniform(45, 95)
            highlight_bright = random.uniform(25, 60)

            # Draw dark furrow groove
            mask_dark = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(mask_dark, [pts], isClosed=False, color=255, thickness=thickness)
            mask_dark = cv2.GaussianBlur(mask_dark, (3, 3), 0)

            # Draw bright displacement burr slightly shifted
            shift_x = int(round(np.cos(angle + np.pi / 2) * (thickness + 1)))
            shift_y = int(round(np.sin(angle + np.pi / 2) * (thickness + 1)))
            shifted_pts = pts + np.array([[[shift_x, shift_y]]], dtype=np.int32)

            mask_bright = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(mask_bright, [shifted_pts], isClosed=False, color=255, thickness=max(1, thickness - 1))
            mask_bright = cv2.GaussianBlur(mask_bright, (3, 3), 0)

            # Apply scratch to canvas
            canvas -= (mask_dark.astype(np.float32) / 255.0) * depth_dark
            canvas += (mask_bright.astype(np.float32) / 255.0) * highlight_bright

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def _inject_crack(self, image: np.ndarray) -> np.ndarray:
        """
        Injects dendritic, jagged crack with branching stress fracture paths.
        """
        h, w = image.shape[:2]
        canvas = image.astype(np.float32)

        start_x = np.random.uniform(0.2 * w, 0.8 * w)
        start_y = np.random.uniform(0.2 * h, 0.8 * h)
        main_angle = np.random.uniform(0, 2 * np.pi)

        branches = [(start_x, start_y, main_angle, random.randint(120, 260), 3)]

        crack_mask = np.zeros((h, w), dtype=np.float32)

        while branches:
            bx, by, b_angle, b_len, b_thick = branches.pop(0)
            pts = [(int(bx), int(by))]
            cx, cy = bx, by
            c_angle = b_angle

            for step in range(b_len):
                # Jagged step
                step_len = np.random.uniform(2.0, 4.0)
                c_angle += np.random.uniform(-0.35, 0.35)
                cx += np.cos(c_angle) * step_len
                cy += np.sin(c_angle) * step_len

                if not (0 <= cx < w and 0 <= cy < h):
                    break
                pts.append((int(cx), int(cy)))

                # Possibility to branch
                if step > 20 and step < b_len - 20 and random.random() < 0.035 and len(branches) < 4:
                    branch_dir = c_angle + np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.1)
                    branches.append((cx, cy, branch_dir, int(b_len * 0.55), max(1, b_thick - 1)))

            if len(pts) > 2:
                poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                branch_canvas = np.zeros((h, w), dtype=np.uint8)
                cv2.polylines(branch_canvas, [poly], isClosed=False, color=255, thickness=b_thick)
                crack_mask = np.maximum(crack_mask, branch_canvas.astype(np.float32) / 255.0)

        crack_mask = cv2.GaussianBlur(crack_mask, (3, 3), 0)
        contrast_drop = np.random.uniform(80, 140)
        canvas -= crack_mask * contrast_drop
        return np.clip(canvas, 0, 255).astype(np.uint8)

    def _inject_patch(self, image: np.ndarray) -> np.ndarray:
        """
        Injects surface patch / inclusion / rolled-in scale with irregular boundary.
        """
        h, w = image.shape[:2]
        canvas = image.astype(np.float32)

        num_patches = random.randint(1, 3)
        for _ in range(num_patches):
            center_x = int(np.random.uniform(0.2 * w, 0.8 * w))
            center_y = int(np.random.uniform(0.2 * h, 0.8 * h))
            radius = np.random.randint(30, 80)

            # Generate random irregular polygon
            num_vertices = random.randint(7, 14)
            angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
            rads = radius * np.random.uniform(0.6, 1.4, num_vertices)

            x_pts = (center_x + rads * np.cos(angles)).astype(np.int32)
            y_pts = (center_y + rads * np.sin(angles)).astype(np.int32)
            pts = np.stack([x_pts, y_pts], axis=-1)

            patch_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(patch_mask, [pts], 255)
            # Smooth irregular outline
            patch_mask = cv2.GaussianBlur(patch_mask, (25, 25), 0)

            mask_norm = patch_mask.astype(np.float32) / 255.0

            # Internal crusty texture
            internal_noise = np.random.normal(-40, 20, (h, w))
            internal_pattern = cv2.GaussianBlur(internal_noise, (7, 7), 0)

            # Oxidation darkening / variation
            intensity_shift = np.random.choice([-60, -45, 55])
            delta = (intensity_shift + internal_pattern) * mask_norm
            canvas += delta

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def _inject_pitting(self, image: np.ndarray) -> np.ndarray:
        """
        Injects pitting / blow-hole porosity: clusters of concave 3D craters.
        """
        h, w = image.shape[:2]
        canvas = image.astype(np.float32)

        cluster_x = np.random.uniform(0.2 * w, 0.8 * w)
        cluster_y = np.random.uniform(0.2 * h, 0.8 * h)
        cluster_spread = np.random.uniform(40, 110)

        num_pits = random.randint(8, 25)

        for _ in range(num_pits):
            px = int(np.random.normal(cluster_x, cluster_spread))
            py = int(np.random.normal(cluster_y, cluster_spread))
            pit_r = random.randint(4, 16)

            if not (pit_r < px < w - pit_r and pit_r < py < h - pit_r):
                continue

            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)

            pit_mask = np.clip(1.0 - (dist / pit_r), 0, 1)
            pit_mask = pit_mask ** 1.5

            # Shadow center
            canvas -= pit_mask * random.uniform(70, 130)

            # Crescent highlight rim on one edge (light specular reflection)
            rim_x = px - int(pit_r * 0.45)
            rim_y = py - int(pit_r * 0.45)
            rim_dist = np.sqrt((x - rim_x) ** 2 + (y - rim_y) ** 2)
            rim_mask = np.clip(1.0 - (rim_dist / (pit_r * 0.6)), 0, 1)
            canvas += rim_mask * random.uniform(30, 65)

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def generate_defective_surface(self) -> np.ndarray:
        """Generates a defective industrial surface with one or more flaw types."""
        # 1. Base metal surface
        base = self.generate_normal_surface()

        # 2. Defect selection
        defect_type = random.choice(["scratch", "crack", "patch", "pitting", "compound"])

        if defect_type == "scratch":
            return self._inject_scratch(base)
        elif defect_type == "crack":
            return self._inject_crack(base)
        elif defect_type == "patch":
            return self._inject_patch(base)
        elif defect_type == "pitting":
            return self._inject_pitting(base)
        else:  # compound defect
            first = random.choice([self._inject_scratch, self._inject_crack])
            second = random.choice([self._inject_patch, self._inject_pitting])
            intermediate = first(base)
            return second(intermediate)

    def generate_dataset(
        self,
        normal_dir: str,
        defective_dir: str,
        num_normal: int = 100,
        num_defective: int = 100,
    ) -> Tuple[int, int]:
        """
        Generates and saves the synthetic benchmark dataset into target directories.
        """
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(defective_dir, exist_ok=True)

        print(f"\n[INFO] Generating {num_normal} normal surface images...")
        for i in tqdm(range(1, num_normal + 1), desc="Normal Samples"):
            img = self.generate_normal_surface()
            filename = f"normal_{i:04d}.png"
            path = os.path.join(normal_dir, filename)
            cv2.imwrite(path, img)

        print(f"\n[INFO] Generating {num_defective} defective surface images...")
        for i in tqdm(range(1, num_defective + 1), desc="Defective Samples"):
            img = self.generate_defective_surface()
            filename = f"defective_{i:04d}.png"
            path = os.path.join(defective_dir, filename)
            cv2.imwrite(path, img)

        return num_normal, num_defective
