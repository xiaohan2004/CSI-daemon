import abc
import socket
import struct
import threading
import numpy as np
import logging
import json
import time
import mysql.connector
from mysql.connector import Error
import configparser

class ReceiveCSI(abc.ABC):
    """
    用于接收CSI数据的基类。
    """
    def __init__(self):
        self.handle_csi = None

    def set_handle_csi(self, handle_csi):
        """
        设置处理CSI数据的回调函数。
        """
        self.handle_csi = handle_csi

    @abc.abstractmethod
    def start(self):
        """
        开始接收CSI数据。
        """
        pass

    @abc.abstractmethod
    def stop(self):
        """
        停止接收CSI数据。
        """
        pass

    def _send_one_csi(self, csi):
        """
        内部接口，用于传递一个CSI数据。
        """
        if self.handle_csi:
            self.handle_csi(csi)


class ReadFromUDP(ReceiveCSI):
    def __init__(self, port, ip='', device_id='device1'):
        super().__init__()
        self.port = port
        self.ip = ip
        self.device_id = device_id
        self.running = False
        # 使用模块名作为 logger 名称
        self.logger = logging.getLogger(__name__)
        self.db_connection = self._connect_to_database()

    def _connect_to_database(self):
        """
        连接到MySQL数据库。
        """
        # 创建配置解析器对象
        config = configparser.ConfigParser()

        # 读取配置文件
        config.read('db_config.ini')

        # 获取数据库连接信息
        db_config = {
            'host': config.get('database', 'host'),
            'user': config.get('database', 'user'),
            'password': config.get('database', 'password'),
            'database': config.get('database', 'database')
        }
        try:
            connection = mysql.connector.connect(
                host = db_config['host'],
                user = db_config['user'],
                password = db_config['password'],
                database = db_config['database']
            )
            return connection
        except Error as e:
            self.logger.error(f"Error connecting to MySQL: {e}")
            return None

    def _insert_csi_data(self, timestamp, csi_data):
        """
        将CSI数据插入数据库。
        """
        if not self.db_connection:
            return

        try:
            cursor = self.db_connection.cursor()
            query = """
            INSERT INTO raw_data (device_id, timestamp, csi_data)
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, (self.device_id, timestamp, json.dumps(csi_data)))
            self.db_connection.commit()
            cursor.close()
        except Error as e:
            self.logger.error(f"Error inserting CSI data: {e}")

    def start(self):
        """
        开始接收CSI数据。
        """
        def receive():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client_address = (self.ip, self.port)
            sock.bind(client_address)
            self.logger.info("Start receiving CSI data...")
            try:
                while self.running:
                    data, address = sock.recvfrom(4096)
                    if data != b'HEART' and len(data) == 132:
                        # 获取当前时间作为时间戳
                        current_timestamp = int(time.time() * 1000)  # 毫秒级时间戳
                        # 解析 CSI 数据
                        csi_raw = struct.unpack('128b', data[4:])
                        csi = np.zeros(64, dtype=np.complex64)
                        for i in range(0, 128, 2):
                            csi[i // 2] = csi_raw[i + 1] + csi_raw[i] * 1j
                        # 将 CSI 数据转换为 JSON 格式
                        csi_json = {
                            'real': [np.real(x).item() for x in csi],
                            'imag': [np.imag(x).item() for x in csi]
                        }
                        # 插入数据库
                        self._insert_csi_data(current_timestamp, csi_json)
                        # 传递给回调函数
                        self._send_one_csi({
                            'timestamp': current_timestamp,
                            'csi': csi
                        })
            finally:
                sock.close()
                if self.db_connection:
                    self.db_connection.close()
                self.logger.info("Stop receiving CSI data.")

        self.running = True
        threading.Thread(target=receive).start()

    def stop(self):
        """
        停止接收CSI数据。
        """
        self.running = False
        self.logger.info("CSI receiver stopped.")


# 示例代码
if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='/tmp/csi_receiver.log',
        filemode='a'
    )

    # 创建ReadFromUDP实例
    receiver = ReadFromUDP(1234, device_id='device1')

    # 设置处理CSI数据的回调函数
    def handle_csi(csi):
        logging.info(f"Received CSI data: {csi}")

    receiver.set_handle_csi(handle_csi)
    # 开始接收CSI数据
    receiver.start()
    # 等待用户输入，用户输入任意字符后停止接收CSI数据
    input()
    # 停止接收CSI数据
    receiver.stop()
    print("CSI receiver stopped.")