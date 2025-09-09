import numpy as np
import cv2 as cv
import sys
import math

from matplotlib import cm
from typing import List

from src.calculations.calculations import get_ventricle_function, get_limits, get_limits_2

class ShortAxis():
    def __init__(
            self, 
            mri, 
            segmentation: np.ndarray=None, 
            normalize: bool=True
        ) -> None:

        self.original_mri = mri
        self.segmentation = segmentation

        self.y_size, self.x_size, self.slice_count, self.phase_count = mri.shape
        self.max_pixel_value, self.min_pixel_value = mri.max(), mri.min()
        if segmentation is not None:
            self.max_pixel_value_segmentation, self.min_pixel_value_segmentation = segmentation.max(), segmentation.min()
        if normalize:
            self.mri = self.trasform_image(mri, mode=['normalize'])
        else:
            self.mri = mri

    def get_slice(self, slice_idx: int, phase_idx: int, transpose:bool=True, type_image:str='original') -> np.ndarray:
        r""" Get a slice of the MRI

        Args:
            slice_idx (int): slice index
            phase_idx (int): phase index
            transpose (bool): transpose the image
            type_image (str): type of image to return

        Returns:
            np.ndarray: slice
        """
        if (type_image == 'segmentation' or type_image == 'overlay') and self.segmentation is None:
            Exception('Segmentation is not available for this image')

        check_indices = self.check_indices_(slice_idx, phase_idx, 0, 0)
        if not check_indices:
            raise Exception('Indices out of bounds')

        if type_image == 'original':
            return self.mri[:,:,slice_idx,phase_idx].T if transpose else self.mri[:,:,slice_idx,phase_idx]
        if type_image == 'segmentation':
            return self.segmentation[:,:,slice_idx,phase_idx].T if transpose else self.segmentation[:,:,slice_idx,phase_idx]
        if type_image == 'overlay':
            mri = self.mri[:,:,slice_idx,phase_idx].T if transpose else self.mri[:,:,slice_idx,phase_idx]
            segmentation = self.segmentation[:,:,slice_idx,phase_idx].T if transpose else self.segmentation[:,:,slice_idx,phase_idx]
            return self.overlay(mri, segmentation)
    

    def get_front_slice(
            self, 
            slice_idx: int, 
            phase_idx: int, 
            transpose: bool=True, 
            type_image: str='original'
        ) -> np.ndarray:
        r""" Get a front slice of the MRI

        Args:
            slice_idx (int): slice index
            phase_idx (int): phase index
            transpose (bool): transpose the image
            type_image (str): type of image to return

        Returns:
            np.ndarray: slice       
        """
        return self.get_slice(slice_idx, phase_idx, transpose, type_image=type_image)

    
    def get_vertical_slice(
            self, 
            x_idx: int, 
            phase_idx: int, 
            ratio: float=1, 
            type_image: str='original'
        ) -> np.ndarray:
        r"""
        Get a transversal slice of the MRI

        Args:
            x_idx (int): x slice index
            phase_idx (int): phase index
            ratio (float): ratio of the image
            type_image (str): type of image to return
        
        Returns:
            np.ndarray: transversal slice
        """
        if (type_image == 'segmentation' or type_image == 'overlay') and self.segmentation is None:
            Exception('Segmentation is not available for this image')

        check_indices = self.check_indices_(0, phase_idx, x_idx, 0)
        if not check_indices:
            raise Exception('Indices out of bounds')

        if type_image == 'original':
            return cv.resize(self.mri[:, x_idx,:, phase_idx], (self.slice_count*ratio, self.y_size), interpolation=cv.INTER_NEAREST_EXACT)
        if type_image == 'segmentation':
            return cv.resize(self.segmentation[:, x_idx,:, phase_idx], (self.slice_count*ratio, self.y_size), interpolation=cv.INTER_NEAREST_EXACT)
        if type_image == 'overlay':
            mri = cv.resize(self.mri[:, x_idx,:, phase_idx], (self.slice_count*ratio, self.y_size), interpolation=cv.INTER_NEAREST_EXACT)
            segmentation = cv.resize(self.segmentation[:, x_idx,:, phase_idx], (self.slice_count*ratio, self.y_size), interpolation=cv.INTER_NEAREST_EXACT)
            return self.overlay(mri, segmentation)
        

    def get_horizontal_slice(
            self, 
            y_idx: int, 
            phase_idx: int, 
            ratio: float=1, 
            type_image: str='original'
        ) -> np.ndarray:
        r"""
        Get a coronal slice of the MRI

        Args:
            y_idx (int): y slice index
            phase_idx (int): hase ipndex
            ratio (float): ratio of the image
            type_image (str): type of image to return
        
        Returns:
            np.ndarray: coronal slice
        """
        if (type_image == 'segmentation' or type_image == 'overlay') and self.segmentation is None:
            Exception('Segmentation is not available for this image')

        check_indices = self.check_indices_(0, phase_idx, 0, y_idx)
        if not check_indices:
            raise Exception('Indices out of bounds')
        
        if type_image == 'original':
            return cv.resize(self.mri[y_idx, :,:, phase_idx], (self.slice_count*ratio, self.x_size), interpolation=cv.INTER_NEAREST_EXACT)
        if type_image == 'segmentation':
            return cv.resize(self.segmentation[y_idx, :,:, phase_idx], (self.slice_count*ratio, self.x_size), interpolation=cv.INTER_NEAREST_EXACT)
        if type_image == 'overlay':
            mri = cv.resize(self.mri[y_idx, :,:, phase_idx], (self.slice_count*ratio, self.x_size), interpolation=cv.INTER_NEAREST_EXACT)
            segmentation = cv.resize(self.segmentation[y_idx, :,:, phase_idx], (self.slice_count*ratio, self.x_size), interpolation=cv.INTER_NEAREST_EXACT)
            return self.overlay(mri, segmentation)
    
    def get_four_chamber_slice(
            self, 
            phase_idx: int, 
            type_image: str='original', 
            perpendicular: bool=False
        ) -> np.ndarray:
        r""" Get the four chamber slice of the MRI

        Args:
            phase_idx (int): phase index
            type_image (str): type of image to return
            perpendicular (bool): if the line is perpendicular to the ventricles
        
        Returns:
            np.ndarray: four chamber slice
        """
        if self.segmentation is None:
            raise Exception('Segmentation is needed for this function')
        
        if type_image == 'overlay':
            raise Exception('Overlay is not available for this function')
        
        height, width, slice_count, phase_count = self.y_size, self.x_size, self.slice_count, self.phase_count

        final_view = []
        center_image = self.get_slice(slice_count // 2, phase_idx=phase_idx, type_image="segmentation", transpose=False)
        vf = get_ventricle_function(center_image, is_transposed=False, perpendicular=perpendicular, cv_format=False)
        if vf is None:
            return None
        m, b = vf
        limits = get_limits_2(height, width, m, b)
        distance = int(math.dist(limits[0], limits[1]))
        line_pixels = list(zip(*np.linspace(*limits, num=distance).astype(int).T))

        for slice_idx in range(slice_count):
            segmentation_slice_image = self.get_slice(slice_idx, phase_idx=phase_idx, transpose=False, type_image=type_image)
            pixel_values = [segmentation_slice_image[y, x] for x, y in line_pixels]
            final_view.append(pixel_values)

        final_view = np.array(final_view)
        return final_view.T, m, b

    # def get_four_chamber_slice_(
    #         self, 
    #         phase_idx: int, 
    #         type_image: str='original', 
    #         perpendicular: bool=False
    #     ) -> np.ndarray:
    #     r""" Get the four chamber slice of the MRI

    #     Args:
    #         phase_idx (int): phase index
    #         type_image (str): type of image to return
    #         perpendicular (bool): if the line is perpendicular to the ventricles
        
    #     Returns:
    #         np.ndarray: four chamber slice
    #     """
    #     if self.segmentation is None:
    #         raise Exception('Segmentation is needed for this function')
        
    #     if type_image == 'overlay':
    #         raise Exception('Overlay is not available for this function')

    #     width, height, slice_count = self.x_size, self.y_size, self.slice_count

    #     center_image = self.get_slice(slice_count // 2, phase_idx, type_image="segmentation", transpose=False)

    #     m, b = get_ventricle_function(center_image, cv_format=True, is_transposed=False, perpendicular=perpendicular)
    #     axis_list = np.arange(0, width) if not perpendicular else np.arange(0, height)
    #     final = []

    #     for slice_idx in range(slice_count):
    #         slice_image = self.get_slice(slice_idx, phase_idx, type_image=type_image, transpose=False)

    #         line_pixels = []
    #         for axis in axis_list:
    #             if not perpendicular:
    #                 y = int(m * axis + b)
    #                 if 0 <= y < height:
    #                     line_pixels.append(slice_image[y, axis])
    #                 else:
    #                     line_pixels.append(0)
    #             else:
    #                 x = int((axis - b) / m)
    #                 if 0 <= x < width:
    #                     line_pixels.append(slice_image[axis, x])
    #                 else:
    #                     line_pixels.append(0)
                

    #         line_pixels = np.array([line_pixels])
    #         final.append(line_pixels)
    #     return np.vstack(final).T, m, b

    def get_volume(
            self, 
            phase_idx: int, 
            type_image: str='original'
        ) -> np.ndarray:
        r""" Get the volume of the MRI
        
        Args:
            phase_idx (int): phase index
            type_image (str): type of image to return
            
        Returns:
            np.ndarray: volume
        """
        if type_image == 'segmentation' and self.segmentation is None:
            raise Exception('Segmentation is not available for this image')

        if phase_idx == -1 and type_image == 'original':
            return self.mri
        elif phase_idx == -1 and type_image == 'segmentation':
            return self.segmentation        

        if type_image == 'original':
            return self.mri[:,:,:,phase_idx]
        elif type_image == 'segmentation':
            return self.segmentation[:,:,:,phase_idx]
        

    def trasform_image(
            self, 
            image: np.ndarray, 
            ratio: float=1, 
            mode: List[str]=['normalize', 'rgb','pil']
        ) -> np.ndarray:
        r""" Transform the image

        Args:
            image (np.ndarray): image
            ratio (float): ratio of the image
            mode (List[str]): modes of the image
        
        Returns:
            np.ndarray: transformed image
        """
        if 'normalize' in mode:
            image = image / (self.max_pixel_value + 1e-6) * 255
            image = image.astype(np.uint8)
        if 'rgb' in mode:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        """
        if 'pil' in mode:
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image, size=(ratio * original_x, ratio * original_y))
        """
        return image
    

    def overlay(
            self, 
            image: np.ndarray, 
            segmentation: np.ndarray, 
            alpha: float=0.2
        ) -> np.ndarray:
        r""" Overlay the image with the segmentation
        
        Args:
            image (np.ndarray): image
            segmentation (np.ndarray): segmentation
            alpha (float): alpha value
        
        Returns:
            np.ndarray: overlayed image
        """
        check_alpha_ = alpha >= 0 and alpha <= 1
        if not check_alpha_:
            raise Exception('Alpha value out of bounds')

        segmentation_exists = np.any(segmentation != 0)

        if not segmentation_exists:
            return cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        
        rgb_image = np.stack([image]*3, axis=-1)
        cmap = cm.jet
        colored_segmentation = cmap(segmentation / self.max_pixel_value_segmentation)[:, :, :3]
        overlay_image = rgb_image / 255.0 * (1 - alpha * segmentation[:, :, np.newaxis]) + alpha * colored_segmentation * segmentation[:, :, np.newaxis]
        return (overlay_image*255).astype(np.uint8)


    def check_indices_(
            self, 
            slice_idx: int, 
            phase_idx: int, 
            x_idx: int, 
            y_idx: int
        ) -> bool:
        r""" Check if the indices are valid

        Args:
            slice_idx (int): slice index
            phase_idx (int): phase index
            x_idx (int): x index
            y_idx (int): y index
        
        Returns:
            bool: if the indices are valid
        """
        if slice_idx >= self.slice_count or slice_idx < 0:
            return False
        if phase_idx >= self.phase_count or phase_idx < 0:
            return False
        if x_idx >= self.x_size or x_idx < 0:
            return False
        if y_idx >= self.y_size or y_idx < 0:
            return False
        return True
    

    def get_label_centroid(
            self, 
            slice_idx: int, 
            phase_idx: int, 
            label: int=1, 
            cv_format: bool=False
        ) -> tuple:
        r""" Get the centroid of a label in a segmented image

        Args:
            slice_idx (int): slice index
            phase_idx (int): phase index
            label (int): label
            cv_format (bool): if the return is in OpenCV coordinates
        
        Returns:
            tuple: centroid coordinates (x, y)
        """
        if self.segmentation is None:
            raise Exception('Segmentation is not available for this image')

        segmentation = self.get_slice(slice_idx, phase_idx, transpose=False, type_image='segmentation')

        y_coords, x_coords = np.where(segmentation == label)

        if len(x_coords) == 0 or len(y_coords) == 0:
            return None  
        
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        if cv_format != False:
            centroid_y = segmentation.shape[1] - centroid_y
        
        return centroid_x, centroid_y
    