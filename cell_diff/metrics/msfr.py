import numpy as np
from frc.deps_types import dip

def compute_2d_psd_fft(image):
    """
    Compute the 2D Power Spectral Density (PSD) using FFT.

    Args:
        image (2D numpy array): Input 2D signal (e.g., an image)

    Returns:
        psd_2d (2D numpy array): 2D power spectral density
        freq_x (1D numpy array): Frequency axis for the x-direction
        freq_y (1D numpy array): Frequency axis for the y-direction
    """
    # Get the dimensions of the image
    ny, nx = image.shape
    
    # 2D FFT and shift the zero frequency component to the center
    fft_2d = np.fft.fft2(image)
    fft_2d_shifted = np.fft.fftshift(fft_2d)
    
    # Compute the 2D Power Spectral Density
    psd_2d = np.abs(fft_2d_shifted) ** 2 / (nx * ny)
    
    # Frequency ranges
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny))
    
    return psd_2d, freq_x, freq_y

def compute_msf_resolution(img, total_intensity=1000.0, image_size=256, threshold=1e-3, pixel_size=320):
    img = total_intensity * img / img.sum()

    psd_2d = compute_2d_psd_fft(img)[0]
    psd_1d = dip.RadialSum(psd_2d, None)
    psd_1d = np.array(psd_1d)

    psd_1d = psd_1d[:image_size // 2]

    index = next((i for i, val in enumerate(psd_1d) if val < threshold), None)

    if index is None:
        index = image_size // 2

    frequency = index / (image_size * pixel_size) 
    resolution = 1 / frequency

    return resolution
