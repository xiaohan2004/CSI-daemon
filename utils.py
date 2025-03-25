import socket

import numpy as np
import scipy.signal
from scipy.ndimage import median_filter


class CsiTime:
    """
    这里的数据是按照时间顺序存储的一个矩阵，每一行是一个时间点的CSI数据。
    唯一特殊的地方就是，每次添加新的CSI数据时（最后一行），会删除最早的CSI数据（第一行）。
    """

    def __init__(self, subcarries_num=64, max_size=64, max_k=9) -> None:
        self.max_k = max_k
        self.max_size = max_size
        self.matrix = np.zeros((max_k * max_size, subcarries_num), dtype=complex)
        self.time = np.zeros(max_k * max_size)
        self.polluted = False

    def append(self, csi):
        self.polluted = True
        remove_first_row = self.matrix[1:]
        self.matrix = np.vstack((remove_first_row, csi))

    def append_with_time(self, csi, tsf):
        remove_first_row = self.matrix[1:]
        self.matrix = np.vstack((remove_first_row, csi))
        self.time = np.hstack((self.time[1:], tsf))

    def get_csi_matrix(self):
        return self.matrix[-self.max_size:]

    def get_corrected_csi_matrix(self):
        assert not self.polluted

        def make_up_list(data, k):
            for i in range(1, len(data)):
                if data[i] <= data[i - 1]:
                    data[i] = data[i - 1] + k
            return data

        # 去除过高的采样率导致的高于10Hz的信号
        size = self.max_size
        time = np.zeros(size)
        k = 0
        while (k == 0 or time[-1] - time[0] < 16 * size) and k < self.max_k:
            k += 1
            matrix = self.matrix[: k * size]
            time = self.time[:k * size]
            time = np.array(time)
            time = (time / 1.0e6)  # 降低时间精度，减少数据量
            time = make_up_list(time, 2 ** 32 / 1e6)

        m = matrix[:, :]
        m = np.where(np.isnan(m), 0, m)
        m = np.apply_along_axis(correct_sampling, 0, m, time, target_interval=16, return_size=size)
        return m

    def try_get_corrected_csi_matrix(self):
        if not self.polluted:
            return self.get_corrected_csi_matrix()
        else:
            return self.get_csi_matrix()


class CSI:
    @staticmethod
    def get_magic_number():
        return [b'\xde\xad\xbe\xef',
                b'\xde\xad\xbe\xaf',
                b'\xbe\xaf\xde\xad']

    @staticmethod
    def find_sub_array_index(array, sub):
        """
        找array中sub所在的第一个位置的第一个下标。

        array: 数组或字节序列，
        sub: 更小的数组或字节序列,

        return: 找到的首个匹配位置下标，否则返回None
        """
        for i in range(0, len(array) - len(sub)):
            if np.array_equal(array[i:i + len(sub)], sub):
                return i
        return None

    @staticmethod
    def find_sub_array_index_rev(array, sub):
        """
        功能和 find_sub_array_index 类似，但是查找顺序是从后面开始的。
        """
        for i in range(len(array) - len(sub), 0, -1):
            if np.array_equal(array[i:i + len(sub)], sub):
                return i
        return None

    @staticmethod
    def get_data_from_pack(pack):
        """
        根据数据开始0xdeadbeef,结束是0xbeafdead，获取到包中的数据部分。
        由于这个两个标记字符的写入为小端序，此函数默认为采用小端序。

        return: 找到的data数据，包含标记序列字符
        """
        # pack.
        magic_number = CSI.get_magic_number()
        start = CSI.find_sub_array_index(pack, magic_number[0][::-1])
        end = CSI.find_sub_array_index_rev(pack, magic_number[2][::-1]) + len(magic_number[2])
        return pack[start:end]

    @staticmethod
    def get_tsf_from_pack(pack):
        """
        从数据包中获取TSF数据，TSF数据是一个4字节的无符号整数。
        """
        data = CSI.get_data_from_pack(pack)
        tsf = data[4:8]
        return int.from_bytes(tsf, byteorder='little')

    @staticmethod
    def get_csi_from_data(data):
        """
        根据csi开始的标记字符是 0xdeadbeaf，和csi数据是512字节长度的条件，进行获取csi数据。

        return: csi的CFR 格式数据，不包括前后标记数据
        """
        magic_number = CSI.get_magic_number()
        # 找到第一个标记数字的字节位置并加上标记长度获得csi数据的开始位置
        start = CSI.find_sub_array_index(data, magic_number[1][::-1]) + len(magic_number[1])

        #: WARN: 为了模拟 data_43012.py 中的行为，start 需要向前移动一个字节
        # start -= 1
        return data[start:start + 512]

    @staticmethod
    def get_csi_vector_from_packet(packet):
        data = CSI.get_data_from_pack(packet)
        csi = CSI.get_csi_from_data(data)
        return CSI.get_vector_from_csi(csi)

    @staticmethod
    def get_rssi_and_channel_info_from_data(data):
        raise NotImplementedError("get_rssi_and_channel_info_from_data is unimplemented.")

    @staticmethod
    def get_vector_from_csi(csi_data, tones_num=64, valid_tone=None):
        """
        csi_data: 从数据包中提取出来的csi字节数据，不包含前后标记数字
        tones_num: 子载波的数量，
        valid_tone: 符合要求的子载波序号，不合法要求的子载波的CFR会设置为nan.
        """
        if valid_tone is None:
            # MIND: 原data_43012.py 写的合法子载波数据范围，原因不知道
            # 这是 802.11n 规定的 52 个子载波的数量。
            valid_tone = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                          25, 26, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                          58, 59, 60, 61, 62, 63];
        # 字节序列转换为numpy数组
        # MIND: 看原脚本使用了int函数转换16进制字符串，其默认是大端序
        # 所有这里也使用大端序的 2字节转换为16位无符号整数
        csi_data = np.frombuffer(csi_data, dtype=">u2")

        ch = np.zeros(tones_num, dtype=complex)
        for i in range(tones_num):
            if i in valid_tone:
                real_number = csi_data[3 * i] & 0x7fff
                real_sign = (csi_data[3 * i] >> 15) & 0x1
                imag_number = csi_data[3 * i + 1] & 0x7fff
                imag_sign = (csi_data[3 * i + 1] >> 15) & 0x1
                # MIND: ch_exp 具体表示意义难以理解，就抄下了
                ch_exp = csi_data[3 * i + 2]
                if ch_exp >= 32:
                    ch_exp = ch_exp - 64
                else:
                    ch_exp = ch_exp
                ch_exp = -3 - ch_exp
                ch_real = real_number * (-2 * real_sign + 1)
                ch_imag = imag_number * (-2 * imag_sign + 1)
                ch[i] = complex(ch_real, ch_imag) * np.exp2(ch_exp)
            else:
                ch[i] = np.nan
        return ch

    @staticmethod
    def get_amplitude(csi):
        return np.abs(csi)

    @staticmethod
    def get_amplitude_db_unit(csi: np.ndarray[complex]):
        # return np.where(np.abs(csi) > 0, 10 * np.log10(np.abs(csi)), np.abs(csi))
        return 20 * np.log10(np.abs(csi) + 0.0000001)

    @staticmethod
    @DeprecationWarning
    def get_phase_old(csi, deg=False):
        return np.angle(csi, deg=deg)

    @staticmethod
    def get_phase(csi):
        return np.angle(csi)

    @staticmethod
    def rebuild_complex(amplitude, phase):
        if phase is None:
            return amplitude
        return amplitude * np.exp(1j * phase)

    @staticmethod
    def get_subcarries_num(csi):
        return len(csi)


class Signal:
    @staticmethod
    def mean_filter(signal, kernel_size, pad_mode: str = 'constant'):
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")

        # 计算填充大小
        pad_size = kernel_size // 2

        # 对信号进行填充
        padded_signal = np.pad(signal, pad_size, mode=pad_mode)

        # 初始化滤波后的信号
        filtered_signal = np.zeros_like(signal)

        # 进行均值滤波
        for i in range(len(signal)):
            filtered_signal[i] = np.mean(padded_signal[i:i + kernel_size])

        return filtered_signal

    @staticmethod
    def median_filter(signal, kernel_size, pad_mode: str = 'constant'):
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")

        # 计算填充大小
        pad_size = kernel_size // 2

        # 对信号进行填充
        padded_signal = np.pad(signal, pad_size, mode=pad_mode)

        # 初始化滤波后的信号
        filtered_signal = np.zeros_like(signal)

        # 进行中值滤波
        for i in range(len(signal)):
            filtered_signal[i] = np.median(padded_signal[i:i + kernel_size])

        return filtered_signal

    @staticmethod
    def hampel_filter(signal, window_size=5, n_sigma=3):
        """
        Apply the Hampel filter to a signal.

        :param signal: The input signal
        :param window_size: The window size for the filter
        :param n_sigma: The number of standard deviations for outlier detection
        :return: The filtered signal
        """
        scale = 1.4826  # 使用正态分布标准偏差估计值（值为1.4826） * 绝对中位差，得到一个估计标准差。
        # Calculate the rolling median and standard deviation
        input_series = signal
        new_series = signal.copy()
        rolling_median = Signal.median_filter(signal, window_size,
                                              pad_mode="reflect")  # signal.rolling(window=window_size, center=True).median()
        difference = np.abs(rolling_median - input_series)

        median_abs_deviation = Signal.median_filter(difference, window_size, pad_mode="reflect")
        threshold = n_sigma * scale * median_abs_deviation
        outliers = difference > threshold
        new_series[outliers] = rolling_median[outliers]
        return new_series


def read_udp_data_txt_to_bytes(file_path):
    """
    为了和data_43012.py 、plot_packaet_43012.py 对比，需要将16进制的字符串文件读取为字节序列，用于后续的处理。
    从txt文件中读取数据，每行数据是16进制的字节数据，返回一个字节序列。

    file_path: 文件路径

    return: 读取到的字节序列,按照packet进行分割的数组
    """
    with open(file_path, "r", encoding='utf-8', errors="replace") as f:
        lines = f.readlines()
    # 分包读取
    packet = [lines[i * 40:i * 40 + 40] for i in range(0, int(len(lines) / 40))]
    # 去掉包内的每行的换行符，空格符
    data = [''.join(p).replace('\n', '').replace(' ', '') for p in packet]
    # 转换为字节序列
    data = [bytes.fromhex(d) for d in data]
    return data


def read_csi_vector_from_csv_file(file_path):
    """
    从csv文件中读取CSI数据，返回一个numpy矩阵。
    """
    data = np.loadtxt(file_path, delimiter=',', dtype=np.complex64)
    return data


def correct_phase1(signal_diff_subcarry):
    ret = np.zeros(signal_diff_subcarry.shape)
    for i in range(signal_diff_subcarry.shape[0]):
        ret[i] = signal_diff_subcarry[i] - signal_diff_subcarry[0]
    return ret


def correct_sampling(signal, actual_times, target_interval=0.1, return_size=10):
    """
    Correct the sampling points of a signal using interpolation.

    :param signal: The original signal values
    :param actual_times: The actual sampling times
    :param target_interval: The target sampling interval (default is 0.1s)
    :return: The corrected signal and the new sampling times
    """
    from scipy.interpolate import interp1d
    # Generate the target times based on the target interval
    target_times = np.arange(actual_times[0], actual_times[-1], target_interval)

    # Create an interpolation function
    # 如果signal 是复数向量，就分别对实部和虚部进行插值
    if np.iscomplexobj(signal):
        interpolation_function_real = interp1d(actual_times, signal.real, kind='linear', fill_value='extrapolate')
        interpolation_function_imag = interp1d(actual_times, signal.imag, kind='linear', fill_value='extrapolate')
        interpolation_function = lambda x: interpolation_function_real(x) + 1j * interpolation_function_imag(x)
    else:
        interpolation_function = interp1d(actual_times, signal, kind='linear', fill_value='extrapolate')

    # Interpolate the signal to the target times
    corrected_signal = interpolation_function(target_times)

    if len(corrected_signal) < return_size:
        print("xxx")

    assert len(corrected_signal) >= return_size
    return corrected_signal[-return_size:]


def correct_amp1(f, alpha=2000, tau=0.0, K=10, DC=0, init=1, tol=1e-5):
    """
    vmd 信号分解算法分解信号并返回图片
    :param f: signal vector
    :param aplha: moderate bandwidth constraint
    :param tau: noise-tolerance (no strict fidelity enforcement)
    :param K: modes number
    :param DC: no DC part imposed
    :param init: initialize omegas uniformly
    :param tol:
    :return: img bytes
    """
    from sktime.libs.vmdpy import VMD
    # 去除直流分量
    f = f - f.mean()

    # Run VMD
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
    ret = []
    limit = 200000
    # ret.append(u[0,:])
    ret.append(u[1, :])
    # ret.append(u[2,:])
    # for i in range(u.shape[0]):
    #     s = u[i,:]
    #     if np.max(s)< limit:
    #         ret.append(s)
    # return np.array(ret).sum(axis=0)
    return np.sum(u, axis=0)


def correct_amp2(signal, cutoff_freq=3, sampling_rate=10):
    """
    Perform FFT on the signal, remove high-frequency components, and return the filtered signal.

    :param signal: Input signal
    :param cutoff_freq: Cutoff frequency to remove high-frequency components
    :param sampling_rate: Sampling rate of the signal
    :return: Filtered signal
    """
    # Perform FFT
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)

    # Remove high-frequency components
    fft_signal[np.abs(freqs) > cutoff_freq] = 0

    # Perform inverse FFT
    filtered_signal = np.fft.ifft(fft_signal)

    return filtered_signal


# def correct_amp3(signal, wavelet='db8', level=1):
#     """
#     Perform wavelet transform on the signal, remove high-frequency components, and return the filtered signal.
#
#     :param signal: Input signal
#     :param wavelet: Type of wavelet to use
#     :param level: Level of decomposition
#     :return: Filtered signal
#     """
#     # Perform wavelet transform
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
#
#     # Thresholding high-frequency components
#     coeffs[1:] = [pywt.threshold(c, value=np.std(c) / 2, mode='soft') for c in coeffs[1:]]
#
#     # Perform inverse wavelet transform
#     filtered_signal = pywt.waverec(coeffs, wavelet)
#
#     return filtered_signal


def correct_phase2(signal_diff_subcarry):
    ret = correct_phase1(signal_diff_subcarry)
    k = ret[-1] / (len(ret) - 1)
    b = np.mean(ret)
    return ret - k * np.arange(len(ret)) - b


def correct_phase3(signal_diff_subcarry):
    ret = correct_phase1(signal_diff_subcarry)
    A = np.vstack([np.arange(len(ret)), np.ones(len(ret))]).T
    k, b = np.linalg.lstsq(A, ret, rcond=None)[0]
    return ret - k * np.arange(len(ret)) - b


def correct_sampling(signal, actual_times, target_interval=0.1, return_size=10):
    """
    Correct the sampling points of a signal using interpolation.

    :param signal: The original signal values
    :param actual_times: The actual sampling times
    :param target_interval: The target sampling interval (default is 0.1s)
    :return: The corrected signal and the new sampling times
    """
    from scipy.interpolate import interp1d
    # Generate the target times based on the target interval
    target_times = np.arange(actual_times[0], actual_times[-1], target_interval)

    # Create an interpolation function
    interpolation_function = interp1d(actual_times, signal, kind='linear', fill_value='extrapolate')

    # Interpolate the signal to the target times
    corrected_signal = interpolation_function(target_times)

    assert len(corrected_signal) >= return_size
    return corrected_signal[-return_size:]


def correct_phase3(signal_diff_subcarry):
    ret = correct_phase1(signal_diff_subcarry)
    A = np.vstack([np.arange(len(ret)), np.ones(len(ret))]).T
    k, b = np.linalg.lstsq(A, ret, rcond=None)[0]
    return ret - k * np.arange(len(ret)) - b


if __name__ == '__main__':
    has_person = read_udp_data_txt_to_bytes("./data/2024-10-07-14_23.restroom.10-31.30minutes.has_people.txt")[:5000]
    no_person = read_udp_data_txt_to_bytes("./data/2024-10-07-15_27.restroom.10-31.10minutes.no_people.txt")[:5000]

    has_person_time = np.array([CSI.get_tsf_from_pack(x) for x in has_person])
    no_person_time = np.array([CSI.get_tsf_from_pack(x) for x in no_person])
    has_person = np.array([CSI.get_csi_vector_from_packet(x) for x in has_person])
    no_person = np.array([CSI.get_csi_vector_from_packet(x) for x in no_person])
    # print(no_person.shape)
    # print(no_person_time.reshape((-1,1)).shape)
    no_person = np.concatenate([no_person_time.reshape((-1, 1)), no_person], axis=1)
    has_person = np.concatenate([has_person_time.reshape((-1, 1)), has_person], axis=1)

    np.savetxt("has_people.csv", has_person, delimiter=",")
    np.savetxt("no_people.csv", no_person, delimiter=",")
