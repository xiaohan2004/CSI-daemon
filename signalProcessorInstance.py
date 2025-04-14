import os
import typing
import numpy as np
import pywt
import scipy.signal
import utils
from abstract import SignalProcessorBase


# 输出原始信号，不做处理
class RawSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        return signal

    @staticmethod
    def get_label() -> str:
        return "raw"

    def get_method_params(self) -> dict:
        return {}


# 均值滤波信号处理算法
class MeanFilterSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        window_size = kwargs.get("window_size", 9)
        if np.iscomplexobj(signal) or signal.dtype == np.complex64:
            amplitude = utils.CSI.get_amplitude(signal)
            angle = utils.CSI.get_phase(signal)
        else:
            amplitude = signal
            angle = None

        amplitude = utils.Signal.mean_filter(
            amplitude, kernel_size=window_size, pad_mode="reflect"
        )
        amplitude = utils.Signal.mean_filter(
            amplitude, kernel_size=3, pad_mode="reflect"
        )
        return utils.CSI.rebuild_complex(amplitude, angle)

    @staticmethod
    def get_label() -> str:
        return "mean_filter"

    def get_method_params(self) -> dict:
        return {
            "window_size": {
                "type": "number",
                "default": 7,
            }
        }


# Hampel滤波器
class HampelFilterSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        window_size = kwargs.get("window_size", 7)
        n_sigmas = 2

        amplitude = utils.CSI.get_amplitude(signal)
        new_series = utils.Signal.hampel_filter(amplitude, window_size, n_sigmas)
        angle = utils.CSI.get_phase(signal)
        return utils.CSI.rebuild_complex(new_series, angle)

    @staticmethod
    def get_label() -> str:
        return "hampel_filter"

    def get_method_params(self) -> dict:
        return {
            "window_size": {
                "type": "number",
                "default": 9,
            }
        }


# Hampel滤波+均值滤波
class HampelAndMeanFilterSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        window_size = kwargs.get("window_size", 11)
        n_sigmas = 2

        amplitude = utils.CSI.get_amplitude(signal)
        new_series = utils.Signal.hampel_filter(amplitude, window_size, n_sigmas)
        new_series = utils.Signal.mean_filter(
            new_series, kernel_size=3, pad_mode="reflect"
        )

        angle = utils.CSI.get_phase(signal)
        return utils.CSI.rebuild_complex(new_series, angle)

    @staticmethod
    def get_label() -> str:
        return "hampel_then_mean_filter"

    def get_method_params(self) -> dict:
        return {
            "window_size": {
                "type": "number",
                "default": 7,
            }
        }


# 傅里叶变换信号处理算法
class FftSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        low_freq = kwargs.get("low_freq", 0.5)
        high_freq = kwargs.get("high_freq", 1000)
        sampling_rate = kwargs.get("sampling_rate", 100)  # 采样率
        signal = MeanFilterSignalProcessor().process(signal)

        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
        fft_signal[(np.abs(freqs) < low_freq) | (np.abs(freqs) > high_freq)] = 0
        return np.fft.ifft(fft_signal)

    @staticmethod
    def get_label() -> str:
        return "fft"

    def get_method_params(self) -> typing.Dict[str, typing.Any]:
        return {
            "low_freq": {
                "type": "number",
                "default": 0.1,
            },
            "high_freq": {
                "type": "number",
                "default": 2,
            },
            "sampling_rate": {
                "type": "number",
                "default": 10,
            },
        }


# 小波变换信号处理算法
class WaveletSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        signal = utils.CSI.get_amplitude(signal)
        wavelet = kwargs.get("wavelet", "db4")
        
        # 小波变换的层数计算方式为： 因为第n层的CD频率范围是[fs/2^(n+1), fs/2^n],CA是[0,fs/2^(n+1)]，n从1开始，
        # 所以为了获取到频率小于freq的信号，层数为log2(fs/freq)-1，其中fs为采样频率，freq为目标信号中的最高频率，一般不超过10Hz
        level = kwargs.get("level", 3)
        mode = kwargs.get("mode", "symmetric")

        coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
        for i in range(1, len(coeffs)):
            coeffs[i] = np.zeros_like(coeffs[i])
        return pywt.waverec(coeffs, wavelet, mode=mode)

    @staticmethod
    def get_label() -> str:
        return "wavelet"

    def get_method_params(self) -> typing.Dict[str, typing.Any]:
        return {
            "wavelet": {
                "type": "string",
                "default": "db4",
            },
            "level": {
                "type": "number",
                "default": 3,
            },
            "mode": {
                "type": "string",
                "default": "symmetric",
            },
        }


# EMD信号处理算法
class EmdSignalProcessor(SignalProcessorBase):
    def process(self, signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        drop_percent = kwargs.get("drop_percent", 0.5)
        spline_kind = kwargs.get("spline_kind", "cubic")
        energy_ratio_thr = kwargs.get("energy_ratio_thr", 0.2)
        std_thr = kwargs.get("std_thr", 0.2)
        svar_thr = kwargs.get("svar_thr", 0.001)
        total_power_thr = kwargs.get("total_power_thr", 0.005)
        range_thr = kwargs.get("range_thr", 0.001)
        extrema_detection = kwargs.get("extrema_detection", "simple")

        from PyEMD import EMD

        emd = EMD(
            spline_kind=spline_kind,
            extrema_detection=extrema_detection,
            energy_ratio_thr=energy_ratio_thr,
            std_thr=std_thr,
            svar_thr=svar_thr,
            total_power_thr=total_power_thr,
            range_thr=range_thr,
        )
        imfs = emd(utils.CSI.get_amplitude(signal))
        drop_size = int(len(imfs) * drop_percent)
        return np.sum(imfs[drop_size:], axis=0)

    @staticmethod
    def get_label() -> str:
        return "emd"

    def get_method_params(self) -> typing.Dict[str, typing.Any]:
        return {
            "drop_percent": {"type": "number", "default": 0.5},
            "spline_kind": {
                "type": "string",
                "default": "cubic",
            },
            "energy_ratio_thr": {
                "type": "number",
                "default": 0.2,
            },
            "std_thr": {
                "type": "number",
                "default": 0.2,
            },
            "svar_thr": {
                "type": "number",
                "default": 0.001,
            },
            "total_power_thr": {
                "type": "number",
                "default": 0.005,
            },
            "range_thr": {
                "type": "number",
                "default": 0.001,
            },
            "extrema_detection": {
                "type": "string",
                "default": "simple",
            },
        }


if __name__ == "__main__":
    # 测试代码
    signal = np.random.randint(0, 100, 64)
    processor = MeanFilterSignalProcessor()
    processed_signal = processor.process(signal)
