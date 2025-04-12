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
    def __init__(self, port, ip="", device_id="device1"):
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
        config.read("db_config.ini")

        # 获取数据库连接信息
        db_config = {
            "host": config.get("database", "host"),
            "user": config.get("database", "user"),
            "password": config.get("database", "password"),
            "database": config.get("database", "database"),
        }
        try:
            connection = mysql.connector.connect(
                host=db_config["host"],
                user=db_config["user"],
                password=db_config["password"],
                database=db_config["database"],
                # 添加以下参数
                autocommit=True,
                buffered=True,
                connection_timeout=30,
                # 设置重连参数
                get_warnings=True,
                raise_on_warnings=True,
                # 添加连接参数
                sql_mode="STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION",
            )
            self.logger.info("Successfully connected to the database.")
            return connection
        except Error as e:
            self.logger.error(f"Error connecting to MySQL: {e}")
            return None

    def _check_connection(self):
        """
        检查数据库连接是否有效，如果无效则重新连接。
        """
        try:
            if self.db_connection is None or not self.db_connection.is_connected():
                self.logger.warning("数据库连接已断开，尝试重新连接...")
                self.db_connection = self._connect_to_database()
                return self.db_connection is not None
            return True
        except Error as e:
            self.logger.error(f"检查数据库连接时发生错误: {e}")
            return False

    def _insert_csi_data(self, timestamp, csi_data):
        """
        将CSI数据插入数据库。
        """
        # 检查并确保数据库连接
        if not self._check_connection():
            self.logger.error("无法插入CSI数据：数据库连接失败")
            return False

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                cursor = self.db_connection.cursor()
                query = """
                INSERT INTO raw_data (device_id, timestamp, csi_data)
                VALUES (%s, %s, %s)
                """
                cursor.execute(query, (self.device_id, timestamp, json.dumps(csi_data)))
                self.db_connection.commit()
                cursor.close()
                return True
            except Error as e:
                retry_count += 1
                self.logger.error(
                    f"插入CSI数据时发生错误 (尝试 {retry_count}/{max_retries}): {e}"
                )

                if retry_count < max_retries:
                    self.logger.info("尝试重新连接数据库...")
                    self.db_connection = self._connect_to_database()
                    time.sleep(1)  # 等待1秒后重试
                else:
                    self.logger.error("达到最大重试次数，放弃插入数据")
                    return False
            except Exception as e:
                self.logger.error(f"插入数据时发生未预期的错误: {e}")
                return False
            finally:
                try:
                    if "cursor" in locals() and cursor is not None:
                        cursor.close()
                except Error as e:
                    self.logger.error(f"关闭游标时发生错误: {e}")

    def start(self):
        """
        开始接收CSI数据。
        """

        def receive():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client_address = (self.ip, self.port)
            sock.bind(client_address)
            self.logger.info("Start receiving CSI data...")

            failed_inserts = 0
            last_receive_time = time.time()  # 记录最后一次接收数据的时间
            received_packets = 0
            start_time = time.time()

            try:
                while self.running:
                    try:
                        # 设置接收超时
                        sock.settimeout(5.0)  # 5秒超时
                        data, address = sock.recvfrom(4096)

                        # 统计接收频率
                        received_packets += 1
                        current_time = time.time()
                        if current_time - start_time >= 10:  # 每10秒输出一次统计
                            rate = received_packets / (current_time - start_time)
                            self.logger.info(
                                f"接收频率: {rate:.2f} 包/秒，共 {received_packets} 包"
                            )
                            received_packets = 0
                            start_time = current_time

                        if data != b"HEART" and len(data) == 132:
                            current_timestamp = int(time.time() * 1000)
                            last_receive_time = time.time()  # 更新最后接收时间

                            # 输出前几个字节用于调试
                            self.logger.debug(f"数据前8字节: {[b for b in data[:8]]}")

                            try:
                                csi_raw = struct.unpack("128b", data[4:])
                                csi = np.zeros(64, dtype=np.complex64)
                                for i in range(0, 128, 2):
                                    csi[i // 2] = csi_raw[i + 1] + csi_raw[i] * 1j

                                csi_json = {
                                    "real": [np.real(x).item() for x in csi],
                                    "imag": [np.imag(x).item() for x in csi],
                                }

                                # 插入数据并检查结果
                                if self._insert_csi_data(current_timestamp, csi_json):
                                    failed_inserts = 0  # 重置失败计数
                                else:
                                    failed_inserts += 1
                                    self.logger.warning(
                                        f"数据库插入失败 {failed_inserts} 次"
                                    )

                                # 传递给回调函数
                                self._send_one_csi(
                                    {"timestamp": current_timestamp, "csi": csi}
                                )

                            except struct.error as e:
                                self.logger.error(f"解析CSI数据时发生错误: {e}")
                            except Exception as e:
                                self.logger.error(f"处理CSI数据时发生未预期的错误: {e}")

                        # 添加其他情况的日志
                        elif data == b"HEART":
                            self.logger.debug("收到心跳包")
                        elif len(data) != 132:
                            self.logger.warning(
                                f"收到非标准长度数据包: {len(data)} 字节"
                            )

                    except socket.timeout:
                        # 接收超时，记录日志但继续运行
                        current_time = time.time()
                        time_since_last_receive = current_time - last_receive_time
                        self.logger.info(
                            f"已经 {time_since_last_receive:.1f} 秒没有收到数据"
                        )
                        continue

                    except socket.error as e:
                        self.logger.error(f"接收数据时发生socket错误: {e}")
                        time.sleep(1)  # 发生错误时等待1秒

                # 线程结束时的统计
                total_time = time.time() - start_time
                if total_time > 0:
                    final_rate = received_packets / total_time
                    self.logger.info(
                        f"最终接收频率: {final_rate:.2f} 包/秒，共 {received_packets} 包"
                    )

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
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="/tmp/csi_receiver.log",
        filemode="a",
    )

    # 创建ReadFromUDP实例
    receiver = ReadFromUDP(1234, device_id="device1")

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
