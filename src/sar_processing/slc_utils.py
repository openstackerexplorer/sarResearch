import numpy as np
import rasterio
from typing import List, Tuple

class SLCUtils:
    """
    Utilities for handling Single Look Complex (SLC) SAR data.
    """
    
    def read_complex_data(self, filepath: str) -> np.ndarray:
        """
        Loads the complex-valued SAR data arrays.
        
        Args:
            filepath: Path to the .tif or .vrt file.
            
        Returns:
            Complex numpy array.
        """
        with rasterio.open(filepath) as src:
            # SLC data usually has two bands: real and imaginary
            # or it is stored as a single complex band
            if src.count == 2:
                real = src.read(1).astype(np.float32)
                imag = src.read(2).astype(np.float32)
                return real + 1j * imag
            elif src.count == 1:
                return src.read(1)
            else:
                raise ValueError(f"Unexpected number of bands in {filepath}: {src.count}")

    def coregister_stack(self, reference_slc: np.ndarray, 
                         secondary_slcs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Aligns a stack of secondary SLCs to a reference SLC.
        
        Args:
            reference_slc: The reference complex image.
            secondary_slcs: List of secondary complex images.
            
        Returns:
            List of coregistered complex images.
        """
        # In a real implementation, this would use cross-correlation 
        # or spectral techniques for sub-pixel alignment.
        # Here we provide a placeholder architecture.
        coregistered = []
        for slc in secondary_slcs:
            # Placeholder for alignment logic
            coregistered.append(slc) 
        return coregistered

    def compute_interferogram(self, slc_1: np.ndarray, slc_2: np.ndarray) -> np.ndarray:
        """
        Calculates the phase difference (interferogram) between two SLC images.
        
        Args:
            slc_1: Reference SLC.
            slc_2: Secondary SLC.
            
        Returns:
            Complex interferogram.
        """
        # Complex conjugate multiplication
        # ifgram = slc1 * slc2*
        return slc_1 * np.conj(slc_2)

def read_complex_data(filepath: str) -> np.ndarray:
    return SLCUtils().read_complex_data(filepath)

def coregister_stack(reference_slc: np.ndarray, secondary_slcs: List[np.ndarray]) -> List[np.ndarray]:
    return SLCUtils().coregister_stack(reference_slc, secondary_slcs)

def compute_interferogram(slc_1: np.ndarray, slc_2: np.ndarray) -> np.ndarray:
    return SLCUtils().compute_interferogram(slc_1, slc_2)
