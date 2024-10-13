import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
from scipy.fft import rfft, irfft

def load_audio(file_path):
    sample_rate, data = wav.read(file_path)
    return sample_rate, data

def clean_and_amplify_audio(input_file, low_cutoff=85, high_cutoff=3000, gain=1.8):
    sample_rate, data = load_audio(input_file)

    # Применение шумоподавления
    reduced_noise = nr.reduce_noise(y=data.astype(np.float32), sr=sample_rate, 
                                    prop_decrease=0.9, stationary=False)

    # Применение оконной функции
    windowed_data = reduced_noise * np.hamming(len(reduced_noise))

    # Применение реального дискретного преобразования Фурье
    fft_data = rfft(windowed_data)

    # Создание маски для частот
    freq_bins = np.fft.rfftfreq(len(windowed_data), d=1/sample_rate)
    mask = (freq_bins >= low_cutoff) & (freq_bins <= high_cutoff)

    # Увеличение амплитуды частот в диапазоне 85-3000 Гц
    fft_data[mask] *= gain

    # Применение обратного преобразования Фурье
    cleaned_data = irfft(fft_data)

    return sample_rate, cleaned_data

