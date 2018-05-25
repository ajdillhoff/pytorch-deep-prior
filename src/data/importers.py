from PIL import Image
import numpy as np


class DepthImporter(object):
    """ Provides general functions for importing raw depth data.

    Attributes:
        fx: focal length in x direction
        fy: focal length in y direction
        ux: principal point in x direction
        uy: principal point in y direction
    """

    def __init__(self, fx, fy, ux, uy):
        """Initialize the DepthImporter object.

        Args:
            fx: focal length in x direction
            fy: focal length in y direction
            ux: principal point in x direction
            uy: principal point in y direction
        """

        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy


class NYUImporter(DepthImporter):
    """ Provides functions related to importing NYU hand pose data.

    Attributes:
        base_path: Path to the NYU dataset base directory
        use_cache: Boolean to indicate if data should be loaded from cache
        cache_path: Path to the stored cache
        num_joints: Number of joints annotated in this dataset
        evaluated_joints: List of joints to be used for evaluation
    """

    def __init__(self, base_path, use_cache=True, cache_path="./cache/"):
        """Initialize the NYUImporter object.

        Args:
            base_path: Path to the NYU dataset base directory
            use_cache: Boolean to indicate if data should be loaded from cache
            cache_path: Path to the stored cache
        """

        super(NYUImporter, self).__init__(588.03, 587.07, 320., 240.)

        self.base_path = base_path
        self.use_cache = use_cache
        self.cache_path = cache_path
        self.num_joints = 36

        # Use joints evaluated in Tompson et al. (2014)
        self.evaluated_joints = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]

    def load_depth_data(self, file_path):
        """Read depth data from file.

        Args:
            file_path: Full path of the raw depth file

        Returns:
            Image data as a numpy array.
        """

        img = Image.open(file_path)
        # top 8 bits of depth are packed into green channel, lower 8 in blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        depth = np.bitwise_or(np.left_shift(g, 8), b)
        img_data = np.asarray(depth, np.float32)

        return img_data
