import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt

def find_local_minima_maxima(series, prominence_threshold=0.1, distance=6):
    """
    Find local minima and maxima in a series.

    Parameters:
    - series: The input series.
    - prominence_threshold: The minimum prominence of peaks, higher values will result in fewer peaks being detected.
    - distance: Minimum horizontal distance in samples between peaks.

    Returns:
    - local_minima: Indices of local minima.
    - local_maxima: Indices of local maxima.
    """
    # Pad the series to include the start and end
    series = np.pad(series, (1, 1), mode='edge')

    # Find peaks
    peaks, _ = find_peaks(series, prominence=prominence_threshold, distance=distance)

    # Find valleys (negative peaks)
    neg_series = -series
    valleys, _ = find_peaks(neg_series, prominence=prominence_threshold, distance=distance)

    # Remove the padding indices
    local_minima = valleys - 1
    local_maxima = peaks - 1

    return local_minima, local_maxima

# Example usage:
if __name__ == "__main__":
    # Replace this with your actual data
    data = np.array([715.2,716.95,716.0,715.1,715.95,715.35,715.5,714.65,713.7,713.95,715.1,714.3,714.15,713.65,714.15,714.15,713.6,713.45,713.8,713.25,713.8,713.8,714.1,711.05,711.45,711.85,711.7,711.5,711.0,711.25,709.0,708.65,709.0,708.15,708.45,708.6,708.8,709.8,709.9,709.35])

    # Find local minima and maxima
    minima, maxima = find_local_minima_maxima(data)

    print("Local Minima:", minima)
    print("Local Maxima:", maxima)
