import os

import numpy as np
import cv2


class HandDetector(object):
    """ Detects the hand in a depth image.

    Assumes that the hand is the closest object to the camera. Uses the Center
    of Mass to crop the hand.

    Attributes:
        img: Depth image to crop from
        max_depth: Maximum depth value
        min_depth: Minimum depth value
        fx: Camera focal length in x dimension
        fy: Camera focal length in y dimension
        importer: Data importer object
    """

    def __init__(self, img, fx, fy, importer=None):
        """Constructor

        Initializes a new HandDetector object.

        Args:
            img: Input depth image
            fx: Camera focal length in x dimension
            fy: Camera focal length in y dimension
            importer: Data importer object
        """

        self.img = img
        self.max_depth = min(1500, img.max())
        self.min_depth = max(10, img.min())

        # Values out of range are 0
        self.img[self.img > self.max_depth] = 0.
        self.img[self.img < self.min_depth] = 0.

        self.fx = fx
        self.fy = fy
        self.importer = importer

    def bounds_from_com(self, com, size):
        """Calculates the boundaries of the crop given the crop size and center
        of mass.

        The values are projected from image space to world space before adding
        the bounding box. The value is then projected back to image space.

        Args:
            com: Center of Mass in mm
            size: 3D bounding box size in mm

        Returns:
            xstart: start of boundary in x dimension
            xend: end of boundary in x dimension
            ystart: start of bounadry in y dimension
            yend: end of bounadry in y dimension
            zstart: start of boundary in z dimension
            zend: end of bounadry in z dimension
        """

        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / self.fx - size[0] / 2.) / com[2] * self.fx))
        xend = int(np.floor((com[0] * com[2] / self.fx + size[0] / 2.) / com[2] * self.fx))
        ystart = int(np.floor((com[1] * com[2] / self.fy - size[1] / 2.) / com[2] * self.fy))
        yend = int(np.floor((com[1] * com[2] / self.fy + size[1] / 2.) / com[2] * self.fy))
        return xstart, xend, ystart, yend, zstart, zend

    def crop_img(self, img, xstart, xend, ystart, yend, zstart, zend, thresh_z=True):
        """Crops the given image using the specified boundaries.

        Args:
            img: Input image
            xstart: Starting value for x-axis bound
            xend: Ending value for x-axis bound
            ystart: Starting value for y-axis bound
            yend: Ending value for y-axis bound
            zstart: Starting value for z-axis bound
            zend: Ending value for z-axis bound
            thresh_z: Boolean to determine if z-values should be thresholded

        Returns:
            A cropped image.
        """

        if len(img.shape) == 2:
            cropped_img = img[max(ystart, 0):min(img.shape[0], yend), max(xstart, 0):min(img.shape[1], xend)].copy()
            # fill in pixels if crop is outside of image
            cropped = np.pad(cropped_img, ((abs(ystart) - max(ystart, 0),
                                            abs(yend) - min(yend, img.shape[0])),
                                           (abs(xstart) - max(xstart, 0),
                                            abs(xend) - min(xend, img.shape[1]))), mode='constant', constant_values=0)
        else:
            raise NotImplementedError()

        if thresh_z is True:
            near_values = np.bitwise_and(cropped_img < zstart, cropped_img != 0)
            far_values = np.bitwise_and(cropped_img > zend, cropped_img != 0)
            cropped_img[near_values] = zstart
            cropped_img[far_values] = 0.

        return cropped_img

    def crop_area_3d(self, com=None, size=(250, 250, 250), img_size=(128, 128)):
        """Performs a 3D crop of the hand.

        Given an input image, a 3D crop centered on the Center of Mass is
        returned.

        Args:
            com: Center of Mass
            size: Size of crop in 3D
            img_size: Output size of cropped image

        Returns:
            A 2D numpy array containing the cropped hand.
        """

        if len(size) != 3:
            raise ValueError("size must be 3D.")

        if len(img_size) != 2:
            raise ValueError("img_size must be 2D")

        if com is None:
            raise ValueError("CoM must be provided.")

        xstart, xend, ystart, yend, zstart, zend = self.bounds_from_com(com, size)

        cropped_img = self.crop_img(self.img, xstart, xend, ystart, yend, zstart, zend)

        # resize to requested image size
        cropped_img = cv2.resize(cropped_img, img_size, interpolation=cv2.INTER_NEAREST)

        return cropped_img
