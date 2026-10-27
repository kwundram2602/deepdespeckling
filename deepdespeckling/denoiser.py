import logging
from typing import Iterator, Tuple

import torch
import numpy as np

from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE


class Denoiser:
    """Class to share parameters beyond denoising functions
    """

    def __init__(self):
        self.device = self.get_device()
        self._models = {}

    def get_model(self, patch_size: int):
        """Get the model for a given patch size, loading its weights only once

        Args:
            patch_size (int): patch size

        Returns:
            model (Model): model loaded with stored weights, in eval mode
        """
        if patch_size not in self._models:
            model = self.load_model(patch_size=patch_size)
            model.eval()
            self._models[patch_size] = model
        return self._models[patch_size]

    def get_device(self) -> str:
        """Get torch device to use depending on gpu's availability

        Returns:
            device (str): device to be used by torch
        """
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        logging.info(f"{device} device is used by torch")

        return device

    def initialize_axis_range(self, image_axis_dim: int, patch_size: int, stride_size: int) -> list:
        """Initialize the convolution range for x or y axis

        Args:
            image_axis_dim (int): axis size
            patch_size (int): patch size
            stride_size (int): stride size

        Returns:
            axis_range (list) : pixel borders of each convolution
        """
        if image_axis_dim == patch_size:
            axis_range = list(np.array([0]))
        else:
            axis_range = list(
                range(0, image_axis_dim - patch_size, stride_size))
            if (axis_range[-1] + patch_size) < image_axis_dim:
                axis_range.extend(
                    range(image_axis_dim - patch_size, image_axis_dim - patch_size + 1))

        return axis_range

    @staticmethod
    def _block_ranges(size: int, block_size: int, margin: int) -> Iterator[Tuple[int, int, int, int]]:
        """Split one axis into blocks read with a margin on both sides

        Yields:
            (read_start, read_stop, write_start, write_stop): the extent to read, and the part
            of it that is kept and written to the output
        """
        for start in range(0, size, block_size):
            stop = min(start + block_size, size)
            yield max(0, start - margin), min(size, stop + margin), start, stop

    def denoise_image_tiled(self, image, out, block_size: int = 2048, patch_size: int = PATCH_SIZE,
                            stride_size: int = STRIDE_SIZE):
        """Denoise an image of any size block by block, so memory is bounded by the block size

        Each block is read with a margin of patch_size pixels on every side and denoised with
        denoise_array; only its inner part is written to out, so the block borders do not show.

        Args:
            image: array-like of shape [height, width] (SAR2SAR) or [height, width, 2] (MERLIN),
                e.g. a numpy memmap; only the blocks are read into memory
            out: array-like of shape [height, width] receiving the denoised amplitude
            block_size (int): size of the blocks written to out. Defaults to 2048.
            patch_size (int): patch size. Defaults to constant PATCH_SIZE.
            stride_size (int): stride size. Defaults to constant STRIDE_SIZE.

        Returns:
            out: the given output array
        """
        height, width = image.shape[0], image.shape[1]
        if height < patch_size or width < patch_size:
            raise ValueError(
                f"image of size {height}x{width} is smaller than the patch size {patch_size}")
        if tuple(out.shape[:2]) != (height, width):
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected ({height}, {width})")

        for r0, r1, w_r0, w_r1 in self._block_ranges(height, block_size, patch_size):
            for c0, c1, w_c0, w_c1 in self._block_ranges(width, block_size, patch_size):
                block = np.asarray(image[r0:r1, c0:c1], dtype=np.float32)
                denoised = self.denoise_array(block, patch_size, stride_size)
                out[w_r0:w_r1, w_c0:w_c1] = denoised[w_r0 - r0:w_r1 - r0, w_c0 - c0:w_c1 - c0]
        return out

    def denoise_array(self, noisy_image: np.array, patch_size: int, stride_size: int) -> np.array:
        """Denoise an image held in memory and return only the denoised amplitude"""
        raise NotImplementedError

    def save_despeckled_images(self):
        raise NotImplementedError

    def denoise_image_kernel(self):
        raise NotImplementedError

    def preprocess_denoised_image(self):
        raise NotImplementedError

    def denoise_image(self):
        raise NotImplementedError

    def denoise_images(self):
        raise NotImplementedError
