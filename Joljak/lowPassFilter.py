import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Apply low-pass-filter
def do_lpf(data_arr: np.ndarray, fs: float, order: int, cutoff: float) -> np.ndarray:
    """
    :param data_arr: data to process
    :param fs: sampling rate
    :param order: n
    :param cutoff: float
    :return: Filtered ndarray
    """
    # Apply low pass filter on VM
    filtered_data = butter_lowpass_filter(data=data_arr, cutoff=cutoff, fs=fs, order=order)
    return filtered_data

if __name__ == '__main__':
    # Filter requirements.
    order = 6
    fs = 30.0  # sample rate, Hz
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response", fontsize=14)
    plt.xlabel('Frequency [Hz]', fontsize=14)
    plt.grid()

    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 5.0  # seconds
    n = int(T * fs)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)

    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]', fontsize=14)
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()