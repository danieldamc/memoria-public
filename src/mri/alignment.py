import numpy as np
import cv2 as cv

from src.mri.shortaxis import ShortAxis
from src.trasformations.utils import shift_array

from scipy.ndimage import gaussian_filter1d

from typing import Tuple

class Alignment(ShortAxis):
    def __init__(
            self, 
            original: np.ndarray, 
            segmentation: np.ndarray, 
            normalize: bool = False, 
            shift_type: str = 'cm', 
            weight_method: str = 's1_priority'
        ) -> None:

        super().__init__(
            original, 
            segmentation, 
            normalize=normalize
        )
        self.aligned_mri = None
        self.aligned_segmentation = None
        self.shift_type = shift_type
        self.weight_method = weight_method
        if self.segmentation is not None:
            self.aligned_mri, self.aligned_segmentation = self._alingment(shift_type, weight_method)


    def _calculate_cm_shifts(
            self, 
            phase_idx: int, 
            label: int = 1, 
            axis:int = 0, 
        ) -> Tuple[np.ndarray, np.ndarray]:
        r""" Calculate the shifts of the image based on the center of mass of the label

        Args:
            phase_idx (int): phase index
            label (int): label to calculate the center of mass
            axis (int): axis if the array to calculate the shift
        """
        #get sagital center slice and calculate the center of mass (cm) of the label
        center_idx = self.slice_count // 2
        x_cm, y_cm = np.array(self.get_label_centroid(center_idx, phase_idx, label=label, cv_format=True), dtype=int)

        #get view according to the axis
        #view = self.get_coronal_slice(x_cm, phase_idx, ratio=1, type_image='segmentation') if axis == 0 else self.get_transversal_slice(y_cm, phase_idx, ratio=1, type_image='segmentation')
        view = self.get_vertical_slice(y_cm, phase_idx, ratio=1, type_image='segmentation') if axis == 0 else self.get_horizontal_slice(x_cm, phase_idx, ratio=1, type_image='segmentation')
        view = np.where(view == label, 1, 0)
        width = self.slice_count

        mask = np.array([1 if np.any(view[:, i] == 1) else 0 for i in range(width)], dtype=bool)
        centers = np.zeros_like(mask, dtype=float)   
        centers = np.array([np.mean(np.where(view[:, i] == 1)) if np.any(view[:, i] == 1) else 0 for i in range(width)])

        mean_centers = np.mean(centers[mask])
        
        mean_centers_arr = np.zeros_like(mask, dtype=float)
        mean_centers_arr[mask] = mean_centers
        shifts = (mean_centers_arr - centers).astype(int)
        return shifts, mask
    
    def _calculate_4ch_cm_shifts(
            self, 
            phase_idx, 
            label = 1,
            perpendicular=False
        ) -> Tuple[np.ndarray, np.ndarray]:
        r""" Calculate the shifts of the image based on TODO

        Args:
            phase_idx (int): phase index
            label (int): label to calculate the center of mass
            axis (int): axis if the array to calculate the shift
        """
        def components(magnitudes, angle):
            y_components = magnitudes * np.sin(angle)
            x_components = magnitudes * np.cos(angle)
            return x_components.astype(int), y_components.astype(int)
        
        if self.segmentation is None:
            raise Exception('Four chamber alignment method is no available if segmentation is not provided')
        
        fcs = self.get_four_chamber_slice(phase_idx, type_image='segmentation', perpendicular=perpendicular)

        if fcs is None:
            return np.array([0] * self.slice_count), np.array([0] * self.slice_count)
        view, m, _ = fcs

        width = self.slice_count

        mask = np.array([1 if np.any(view[:, i] == label) else 0 for i in range(width)], dtype=bool)
        centers = np.zeros_like(mask, dtype=float)   
        centers = np.array([np.mean(np.where(view[:, i] == label)) if np.any(view[:, i] == label) else 0 for i in range(width)])

        center_slice_idx = width // 2

        reference_center = centers[center_slice_idx]
        mean_centers_arr = np.zeros_like(mask, dtype=float)
        mean_centers_arr[mask] = reference_center
        shifts = (mean_centers_arr - centers).astype(int)
        return components(shifts, np.arctan(m))
    
    def _calculate_xcorr_shifts(
            self,
            phase_idx: int,
            axis: int = 0,
            gaussian_sigma: int = None
        ) -> list:
        r""" Calculate the shifts of the image using cross-correlation across the center 
        of mass of the left ventricle

        Args:
            phase_idx (int): phase index
            label (int): label to calculate the center of mass
            axis (int): axis if the array to calculate the shift
        """
        slice_count = self.slice_count
        center_slice_idx = slice_count // 2

        x_centroid, y_centroid = [int(coordinate) for coordinate in self.get_label_centroid(slice_idx=center_slice_idx, phase_idx=phase_idx, label=1, cv_format=False)]
        if axis == 0:
            view = self.get_vertical_slice(x_centroid, phase_idx=phase_idx, ratio=1, type_image='original') # axis 0 => vertical slice => vertival movement
        elif axis == 1:
            view = self.get_horizontal_slice(y_centroid, phase_idx=phase_idx, ratio=1, type_image='original') # axis 1 => horizontal slice => horizontal movement

        shifts = []
        last = 0
        for i in range(slice_count - 1):
            signal_i = view[:, i]
            signal_i_plus_1 = view[:, i + 1]

            if gaussian_sigma is not None:
                signal_i = gaussian_filter1d(signal_i, sigma=gaussian_sigma)
                signal_i_plus_1 = gaussian_filter1d(signal_i_plus_1, sigma=gaussian_sigma)
        
            correlation = np.correlate(signal_i, signal_i_plus_1, mode='full')
            lags = np.arange(-len(signal_i) + 1, len(signal_i))
            max_corr_index = np.argmax(correlation)
            max_corr_lag = lags[max_corr_index]
            
            shift = max_corr_lag + last
            shifts.append(shift)
            last = shift
        shifts.insert(0, 0)
        shifts = list(np.array(shifts) - (shifts[center_slice_idx]))
        return np.array(shifts) if axis == 1 else -np.array(shifts)
    
    def _calculate_4ch_xcorr_shifts(
            self,
            phase_idx: int,
            axis: int = 0,
            gaussian_sigma: int = None
        ) -> list:
        r""" Calculate the shifts of the image using cross-correlation across the center
        
        Args:
            phase_idx (int): phase index
            axis (int): axis if the array to calculate the shift
            gaussian_sigma (int): sigma for the gaussian filter
        
        Returns:
            list: shifts
        """
        def components(magnitudes, angle):
            y_components = magnitudes * np.sin(angle)
            x_components = magnitudes * np.cos(angle)
            return x_components.astype(int), y_components.astype(int)

        slice_count = self.slice_count
        center_slice_idx = slice_count // 2

        if axis == 0:
            fcs = self.get_four_chamber_slice(phase_idx=phase_idx, type_image='original', perpendicular=False)
        elif axis == 1:
            fcs = self.get_four_chamber_slice(phase_idx=phase_idx, type_image='original', perpendicular=True)
        
        if fcs is None:
            return components(np.array([0] * slice_count), np.array([0] * slice_count))
        view, m, _ = fcs

        shifts = []
        last = 0
        for i in range(slice_count - 1):
            signal_i = view[:, i]
            signal_i_plus_1 = view[:, i + 1]

            if gaussian_sigma is not None:
                signal_i = gaussian_filter1d(signal_i, sigma=gaussian_sigma)
                signal_i_plus_1 = gaussian_filter1d(signal_i_plus_1, sigma=gaussian_sigma)
        
            correlation = np.correlate(signal_i, signal_i_plus_1, mode='full')
            lags = np.arange(-len(signal_i) + 1, len(signal_i))
            max_corr_index = np.argmax(correlation)
            max_corr_lag = lags[max_corr_index]
            
            shift = max_corr_lag + last
            shifts.append(shift)
            last = shift
        shifts.insert(0, 0)
        shifts = list(np.array(shifts) - (shifts[center_slice_idx]))
        return components(np.array(shifts), np.arctan(m))
    
    def _weight_shifts(
            self,
            s1: Tuple[np.ndarray, np.ndarray], 
            s2: Tuple[np.ndarray, np.ndarray], 
            method='s1_priority'
        ) -> np.ndarray:
        r""" Function to weight the horizontal and vertical shifts according to a weight method

        Args:
            s1 (Tuple[np.ndarray, np.ndarray]): shift and mask of the first view
            s2 (Tuple[np.ndarray, np.ndarray]): shift and mask of the second view
            method (str): method to weight the shifts
        
        Returns:
            np.ndarray: weighted shifts
        """
        shift_1, mask_1 = s1
        shift_2, mask_2 = s2

        if method == 's1': return shift_1
        elif method == 's2': return shift_2
        elif method == 'absmax':
            return np.where(np.abs(shift_1) > np.abs(shift_2), shift_1, shift_2)
        elif method == 'mean':
            return (shift_1 + shift_2) // 2
        elif method == 's1_priority':
            final_shift = np.zeros_like(shift_1, dtype=int)
            final_shift[mask_1] = shift_1[mask_1]
            final_shift[np.bitwise_xor(mask_1, mask_2)] = shift_2[np.bitwise_xor(mask_1, mask_2)]
            return final_shift
        

    def _get_shifts(
            self, 
            phase_idx: int, 
            shift_type: str='cm', 
            weight_method: str='s1_priority'
        ) -> Tuple[np.ndarray, np.ndarray]:
        r""" Get the shifts of the image
        
        Args:
            phase_idx (int): phase index
            shift_type (str): type of shift
            weight_method (str): method to weight the shifts

        Returns:
            Tuple[np.ndarray, np.ndarray]: vertical and horizontal shifts    
        """
        if shift_type == 'cm':
            vertical_shift = self._weight_shifts(
                self._calculate_cm_shifts(phase_idx, label=1, axis=0), 
                self._calculate_cm_shifts(phase_idx, label=2, axis=0), 
                method=weight_method
            )
            horizontal_shift = self._weight_shifts(
                self._calculate_cm_shifts(phase_idx, label=1, axis=1), 
                self._calculate_cm_shifts(phase_idx, label=2, axis=1), 
                method=weight_method
            )
            return vertical_shift, horizontal_shift
        elif shift_type == 'xcorr':
            vertical_shift = self._calculate_xcorr_shifts(phase_idx, axis=0)
            horizontal_shift = self._calculate_xcorr_shifts(phase_idx, axis=1)
            return vertical_shift, horizontal_shift
        elif shift_type == '4ch_cm':
            x_axis_0, y_axis_0 = self._calculate_4ch_cm_shifts(phase_idx, label=1, perpendicular=False) 
            x_axis_1, y_axis_1 = self._calculate_4ch_cm_shifts(phase_idx, label=1, perpendicular=True)
            vertical_shift_1 = x_axis_0 + x_axis_1
            horizontal_shift_1 = y_axis_0 + y_axis_1

            # x_axis_0, y_axis_0 = self._calculate_4ch_cm_shifts(phase_idx, label=2, perpendicular=False) 
            # x_axis_1, y_axis_1 = self._calculate_4ch_cm_shifts(phase_idx, label=2, perpendicular=True)
            # vertical_shift_2 = x_axis_0 + x_axis_1
            # horizontal_shift_2 = y_axis_0 + y_axis_1

            # vertical_shift = self._weight_shifts(
            #     vertical_shift_1, 
            #     vertical_shift_2, 
            #     method=weight_method
            # )
            # horizontal_shift = self._weight_shifts(
            #     horizontal_shift_1, 
            #     horizontal_shift_2, 
            #     method=weight_method
            # )
            return vertical_shift_1, horizontal_shift_1
        
        elif shift_type == '4ch_xcorr':
            x_axis_0, y_axis_0 = self._calculate_4ch_xcorr_shifts(phase_idx=phase_idx, axis=0)
            x_axis_1, y_axis_1 = self._calculate_4ch_xcorr_shifts(phase_idx=phase_idx, axis=1)

            vertical_shift = x_axis_0 + x_axis_1
            horizontal_shift = y_axis_0 + y_axis_1
            return -vertical_shift, -horizontal_shift
        else:
            raise Exception('Shift type not available')

            
    def _alingment(
            self, 
            shift_type= 'cm', 
            weight_method= 's1_priority'
        ) -> Tuple[np.ndarray, np.ndarray]:
        r""" Aling the MRI

        Args:
            shift_type (str): type of shift
            weight_method (str): method to weight the shifts
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: aligned mri and segmentation
        """
        final_image = np.zeros((self.y_size, self.x_size, self.slice_count, self.phase_count))
        final_segmentation = np.zeros((self.y_size, self.x_size, self.slice_count, self.phase_count))

        for phase_idx in range(self.phase_count):
            vertical_shift, horizontal_shift = self._get_shifts(phase_idx, shift_type=shift_type, weight_method=weight_method)

            for slice_idx in range(self.slice_count):
                image_ = self.get_slice(slice_idx, phase_idx, transpose=False, type_image='original')
                segmentation_ = self.get_slice(slice_idx, phase_idx, transpose=False, type_image='segmentation')

                final_image[:,:,slice_idx, phase_idx] = shift_array(shift_array(image_, vertical_shift[slice_idx], axis=0), horizontal_shift[slice_idx], axis=1)
                final_segmentation[:,:, slice_idx, phase_idx] = shift_array(shift_array(segmentation_, vertical_shift[slice_idx], axis=0), horizontal_shift[slice_idx], axis=1)
        
        return final_image, final_segmentation
        #return final_image.astype(np.uint8), final_segmentation.astype(np.uint8)
    

    def get_slice(self, 
            slice_idx: int, 
            phase_idx: int, 
            transpose: bool=True, 
            type_image: str='original', 
            aligned: bool=False
        ) -> np.ndarray:
        r""" Get a slice of the MRI

        Args:
            slice_idx (int): slice index
            phase_idx (int): phase index
            transpose (bool): transpose the image
            type_image (str): type of image to return
            aligned (bool): boolean to get the aligned image

        Returns:
            np.ndarray: slice
        """
        if aligned:
            volume = self.aligned_mri
            segmentation_volume = self.aligned_segmentation
        else:
            volume = self.mri
            segmentation_volume = self.segmentation

        if (type_image == 'segmentation' or type_image == 'overlay') and segmentation_volume is None:
            Exception('Segmentation is not available for this image')

        check_indices = self.check_indices_(slice_idx, phase_idx, 0, 0)
        if not check_indices:
            raise Exception('Indices out of bounds')

        if type_image == 'original':
            return volume[:,:,slice_idx,phase_idx].T if transpose else volume[:,:,slice_idx,phase_idx]
        if type_image == 'segmentation':
            return segmentation_volume[:,:,slice_idx,phase_idx].T if transpose else segmentation_volume[:,:,slice_idx,phase_idx]
        if type_image == 'overlay':
            mri = volume[:,:,slice_idx,phase_idx].T if transpose else volume[:,:,slice_idx,phase_idx]
            segmentation = segmentation_volume[:,:,slice_idx,phase_idx].T if transpose else segmentation_volume[:,:,slice_idx,phase_idx]
            return self.overlay(mri, segmentation)


    def get_front_slice(
            self, 
            slice_idx: int, 
            phase_idx: int, 
            transpose: bool=True, 
            type_image: str='original', 
            aligned: bool=False
        ) -> np.ndarray:
        r""" Get a sagital slice of the MRI

        Args:
            slice_idx (int): slice index
            phase_idx (int): phase index
            transpose (bool): transpose the image
            type_image (str): type of image to return
            aligned (bool): boolean to get the aligned image

        Returns:
            np.ndarray: slice       
        """
        return self.get_slice(slice_idx, phase_idx, transpose, type_image=type_image, aligned=aligned)
    

    def get_vertical_slice(
            self, 
            x_idx: int, 
            phase_idx: int, 
            ratio: float=1, 
            type_image: str='original', 
            aligned: bool=False
        ) -> np.ndarray:
        r""" Get a transversal slice of the MRI

        Args:
            x_idx (int): x slice index
            phase_idx (int): phase index
            ratio (float): ratio of the image
            type_image (str): type of image to return
            aligned (bool): boolean to get the aligned image
        
        Returns:
            np.ndarray: transversal
        """
        if aligned:
            volume = self.aligned_mri
            segmentation_volume = self.aligned_segmentation
        else:
            volume = self.mri
            segmentation_volume = self.segmentation

        if (type_image == 'segmentation' or type_image == 'overlay') and segmentation_volume is None:
            Exception('Segmentation is not available for this image')

        check_indices = self.check_indices_(0, phase_idx, x_idx, 0)
        if not check_indices:
            raise Exception('Indices out of bounds')

        if type_image == 'original':
            return cv.resize(volume[:, x_idx,:, phase_idx], (self.slice_count*ratio, self.y_size), interpolation=cv.INTER_NEAREST_EXACT)
        if type_image == 'segmentation':
            return cv.resize(segmentation_volume[:, x_idx,:, phase_idx], (self.slice_count*ratio, self.y_size), interpolation=cv.INTER_NEAREST_EXACT)
        if type_image == 'overlay':
            mri = cv.resize(volume[:, x_idx,:, phase_idx], (self.slice_count*ratio, self.y_size), interpolation=cv.INTER_NEAREST_EXACT)
            segmentation = cv.resize(segmentation_volume[:, x_idx,:, phase_idx], (self.slice_count*ratio, self.y_size), interpolation=cv.INTER_NEAREST_EXACT)
            return self.overlay(mri, segmentation)
        

    def get_horizontal_slice(
            self, 
            y_idx: int, 
            phase_idx: int, 
            ratio: float=1, 
            type_image: str='original', 
            aligned: bool=False
        ) -> np.ndarray:
        r""" Get a horizontal slice of the MRI

        Args:
            y_idx (int): y slice index
            phase_idx (int): phase index
            ratio (float): ratio of the image
            type_image (str): type of image to return
            aligned (bool): boolean to get the aligned image
        
        Returns:
            np.ndarray: coronal
        """
        if aligned:
            volume = self.aligned_mri
            segmentation_volume = self.aligned_segmentation
        else:
            volume = self.mri
            segmentation_volume = self.segmentation

        if (type_image == 'segmentation' or type_image == 'overlay') and segmentation_volume is None:
            Exception('Segmentation is not available for this image')


        check_indices = self.check_indices_(0, phase_idx, 0, y_idx)
        if not check_indices:
            raise Exception('Indices out of bounds')
        
        if type_image == 'original':
            return cv.resize(volume[y_idx, :,:, phase_idx], (self.slice_count*ratio, self.x_size), interpolation=cv.INTER_NEAREST_EXACT)
        if type_image == 'segmentation':
            return cv.resize(segmentation_volume[y_idx, :,:, phase_idx], (self.slice_count*ratio, self.x_size), interpolation=cv.INTER_NEAREST_EXACT)
        if type_image == 'overlay':
            mri = cv.resize(volume[y_idx, :,:, phase_idx], (self.slice_count*ratio, self.x_size), interpolation=cv.INTER_NEAREST_EXACT)
            segmentation = cv.resize(segmentation_volume[y_idx, :,:, phase_idx], (self.slice_count*ratio, self.x_size), interpolation=cv.INTER_NEAREST_EXACT)
            return self.overlay(mri, segmentation)
        
    def realingment(
            self, 
            shift_type: str='cm', 
            weight_method: str='s1_priority'
        ) -> Tuple[np.ndarray, np.ndarray]:
        r""" Realignment of the MRI
        
        Args:
            shift_type (str): type of shift
            weight_method (str): method to weight the shifts

        Returns:
            Tuple[np.ndarray, np.ndarray]: aligned mri and segmentation
        """
        self.aligned_mri, self.aligned_segmentation = self._alingment(shift_type, weight_method)
        return self.aligned_mri, self.aligned_segmentation
    
    def get_volume(self, type_image:str='original', aligned:bool=False) -> np.ndarray:
        r""" Get the volume of the MRI

        Args:
            type_image (str): type of image to return
            aligned (bool): boolean to get the aligned image
        
        Returns:
            np.ndarray: volume
        """
        if aligned:
            mri = self.aligned_mri
            segmentation = self.aligned_segmentation
        else:
            mri = self.mri
            segmentation = self.segmentation

        if type_image == 'original':
            return mri
        elif type_image == 'segmentation':
            return segmentation
        elif type_image == 'overlay':
            raise Exception('Overlay not available for volume')
            #return self.overlay(mri, self.segmentation)
        else:
            raise Exception(f'{type_image} not available')
        
    
    def get_label_centroid(
            self, 
            slice_idx: int, 
            phase_idx: int, 
            label: int=1,
            aligned: bool=False,
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

        segmentation = self.get_slice(slice_idx, phase_idx, transpose=False, type_image='segmentation', aligned=aligned)

        y_coords, x_coords = np.where(segmentation == label)

        if len(x_coords) == 0 or len(y_coords) == 0:
            return None  
        
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        if cv_format != False:
            centroid_y = segmentation.shape[1] - centroid_y
        
        return centroid_x, centroid_y