import abc
import socket
import struct
import threading

# numpy 是该python文件唯一的一个第三方库
import numpy as np


class ReceiveCSI(abc.ABC):
    """
    用于接收CSI数据的基类。
    基本使用方法是：
    1. 创建一个ReceiveCSI的实现子类的实例
    2. 设置处理CSI数据的回调函数
    3. 调用start方法开始接收CSI数据
    4. 调用stop方法停止接收CSI数据
    """

    def __init__(self):
        self.handle_csi = None

    def set_handle_csi(self, handle_csi):
        """
        设置处理CSI数据的回调函数。
        给外部调用，用于设置处理CSI数据的回调函数。
        :param handle_csi: 处理CSI数据的回调函数，函数原型为handle_csi(csi)，csi为CSI数据，
            csi的数据为一个字典，包含如下键值：{
            timestamp: int, 时间戳，单位为microsecond
            csi: np.array, CSI数据，为一个一维数组，形状为(子载波数量[64]), 数据类型为complex64
            }
        """
        self.handle_csi = handle_csi

    @abc.abstractmethod
    def start(self):
        """
        开始接收CSI数据，给方法实现子类实现。要求不能阻塞线程。可以使用多线程。
        """
        pass

    @abc.abstractmethod
    def stop(self):
        """
        停止接收CSI数据，给方法实现子类实现。要求不能阻塞线程。
        """
        pass

    def _send_one_csi(self, csi):
        """
        内部接口，用于提供给实现子类调用，传递一个CSI数据
        """
        self.handle_csi(csi)


class ReadFromUDP(ReceiveCSI):
    def __init__(self, port, ip=''):
        """
        port: int, 接收CSI数据的端口号
        ip: str, 接收CSI数据的本机IP地址，主要针对多网卡设备。默认为空字符串，一般不需要设置。
        """
        super().__init__()
        self.port = port
        self.ip = ip
        self.running = False

    def start(self):
        """
        开始接收CSI数据
        """

        def receive():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client_address = (self.ip, self.port)
            sock.bind(client_address)
            print("start receiving")
            try:
                while self.running:
                    data, address = sock.recvfrom(4096)
                    if data != b'HEART' and len(data) == 132:
                        timestamp = struct.unpack('<I', data[:4])[0]
                        csi_raw = struct.unpack('128b', data[4:])
                        csi = np.zeros(64, dtype=complex)
                        for i in range(0, 128, 2):
                            csi[i // 2] = csi_raw[i + 1] + csi_raw[i] * 1j
                        self._send_one_csi({
                            'timestamp': timestamp,
                            'csi': csi
                        })
            finally:
                sock.close()
                print("stop receiving")

        self.running = True
        threading.Thread(target=receive).start()

    def stop(self):
        """
        停止接收CSI数据
        """
        self.running = False


# 接受数据中，只有这些子载波是合适的，其他下标的子载波不需要使用。
LLTF_VALID_INDEX = [i for i in range(6, 32)] + [i for i in range(33, 59)]  # 52个合法子载波

# 以下是一个使用ReadFromUDP的示例，用于接收CSI数据并打印CSI数据
# 如果wireshark可以抓取到UDP数据包，但是该代码无法接收到CSI数据
# 可以尝试将本机的防火墙关闭，参考连接：https://blog.csdn.net/qq_36257015/article/details/128965285
if __name__ == '__main__':
    # 创建一个ReadFromUDP的实例，端口号为12345
    receiver = ReadFromUDP(1234)


    # 设置处理CSI数据的回调函数
    def handle_csi(csi):
        print(csi)
        raw_csi = np.copy(csi["csi"])
        INVALID_INDEX = [i for i in range(0, 64) if i not in LLTF_VALID_INDEX]
        raw_csi[INVALID_INDEX] = 0  # 将无效的子载波的CSI数据置为0
        print("csi 信号:", raw_csi)


    receiver.set_handle_csi(handle_csi)
    # 开始接收CSI数据
    receiver.start()
    # 等待用户输入，用户输入任意字符后停止接收CSI数据
    input()
    # 停止接收CSI数据
    receiver.stop()
    print("stop")
