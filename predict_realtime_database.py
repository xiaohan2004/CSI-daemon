import torch
import numpy as np
import logging
import time
import configparser
import mysql.connector
from mysql.connector import Error
from receiveCSI import ReadFromUDP, LLTF_VALID_INDEX
from support_web import (
    create_signal2features_preprocess,
    create_model,
    get_last_dim_size,
    create_preprocess_chain,
    single_signal_preprocess_to_matrix_preprocess,
    get_class_by_label,
    find_implementations,
)
from abstract import SignalProcessorBase
from models import support


class RealtimePredictorDatabase:
    def __init__(self, device_id="device1"):
        """
        初始化实时预测器，带数据库存储功能

        参数:
            device_id: 设备ID，用于数据库记录
        """
        self.device_id = device_id
        self.logger = logging.getLogger(__name__)
        self.is_predicting = False
        self.signal_buffer = []
        self.db_connection = self._connect_to_database()

        # 读取预测配置
        config = self._load_predictor_config()

        # 从配置中获取参数
        self.window_size = support.LEN_W
        self.model_name = config["model_name"]
        self.model_path = config["model_path"]
        self.signal_process_method = config["signal_process_method"]
        self.feature_type = config["feature_type"]
        self.udp_port = config["udp_port"]

        # 初始化UDP接收器
        self.receiver = ReadFromUDP(self.udp_port)

        # 加载模型
        self.model = create_model(
            self.model_name, get_last_dim_size(self.feature_type, 64)
        )
        self.model.get_model().load_state_dict(
            torch.load(self.model_path, weights_only=True)
        )
        self.model.get_model().eval()

        # 设置信号处理和特征提取
        signal_processor_methods = find_implementations(SignalProcessorBase)
        denoise_process_cls = get_class_by_label(
            signal_processor_methods, self.signal_process_method
        )
        denoise_preprocesses = (
            [denoise_process_cls()] if denoise_process_cls is not None else []
        )
        denoise_preprocesses = [
            single_signal_preprocess_to_matrix_preprocess(x)
            for x in denoise_preprocesses
        ]

        # 特征提取
        features_preprocess = create_signal2features_preprocess(self.feature_type)
        self.preprocess = create_preprocess_chain(
            denoise_preprocesses + [features_preprocess]
        )

        # 记录时间戳
        self.start_timestamp = None
        self.last_prediction_time = 0

        self.logger.info(f"初始化RealtimePredictorDatabase完成: {config}")

    def _load_predictor_config(self):
        """从配置文件加载预测器配置"""
        config_parser = configparser.ConfigParser()
        config_parser.read("predictor_config.ini")

        config = {
            "model_name": config_parser.get("predictor", "model_name"),
            "model_path": config_parser.get("predictor", "model_path"),
            "signal_process_method": config_parser.get(
                "predictor", "signal_process_method"
            ),
            "feature_type": config_parser.get("predictor", "feature_type"),
            "udp_port": int(
                config_parser.get("predictor", "udp_port", fallback="1234")
            ),
        }
        return config

    def _connect_to_database(self):
        """连接到MySQL数据库"""
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
                autocommit=True,
                buffered=True,
                connection_timeout=30,
                consume_results=True,
                sql_mode="STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION",
            )

            # 设置会话变量
            cursor = connection.cursor()
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
            cursor.execute("SET SESSION innodb_lock_wait_timeout=50")
            cursor.close()

            self.logger.info("成功连接到数据库")
            return connection
        except Error as e:
            self.logger.error(f"连接MySQL时出错: {e}")
            return None

    def _check_connection(self):
        """检查数据库连接是否有效，如果无效则重新连接"""
        try:
            if self.db_connection is None or not self.db_connection.is_connected():
                self.logger.warning("数据库连接已断开，尝试重新连接...")
                self.db_connection = self._connect_to_database()
                return self.db_connection is not None
            return True
        except Error as e:
            self.logger.error(f"检查数据库连接时发生错误: {e}")
            return False

    def handle_csi(self, csi_data):
        """处理接收到的CSI数据"""
        if not self.is_predicting:
            return

        # 记录开始时间戳（如果尚未设置）
        if self.start_timestamp is None:
            self.start_timestamp = int(time.time() * 1000)  # 毫秒时间戳

        # 获取CSI数据并处理无效子载波
        raw_csi = np.copy(csi_data["csi"])
        invalid_index = [i for i in range(0, 64) if i not in LLTF_VALID_INDEX]
        raw_csi[invalid_index] = 0

        # 更新信号缓冲区
        self.signal_buffer.append(raw_csi)
        if len(self.signal_buffer) > self.window_size:
            self.signal_buffer.pop(0)

        # 当累积足够的数据时进行预测
        if len(self.signal_buffer) == self.window_size:
            current_time = time.time()

            # 每2秒进行一次预测并存储到数据库
            if current_time - self.last_prediction_time >= 2:
                signal_matrix = np.array(self.signal_buffer)

                # 预处理信号
                processed_signal = self.preprocess(signal_matrix)
                processed_signal = processed_signal.reshape(1, *processed_signal.shape)

                # 转换为tensor并预测
                tensor_signal = torch.tensor(processed_signal).float()
                with torch.no_grad():
                    prediction = self.model.get_model()(tensor_signal).numpy()

                # 获取预测结果
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]

                # 记录时间戳
                end_timestamp = int(time.time() * 1000)  # 毫秒时间戳

                # 输出预测结果到日志
                if predicted_class == 0:
                    self.logger.info(
                        f"\033[91m预测类别: 0, 置信度: {confidence:.4f}\033[0m"
                    )
                elif predicted_class == 1:
                    self.logger.info(
                        f"\033[92m预测类别: 1, 置信度: {confidence:.4f}\033[0m"
                    )

                # 将数据插入数据库
                self._insert_status_data(
                    self.start_timestamp, end_timestamp, predicted_class, confidence
                )

                # 更新时间戳
                self.start_timestamp = end_timestamp
                self.last_prediction_time = current_time

    def _insert_status_data(self, start_timestamp, end_timestamp, status, confidence):
        """将状态数据插入数据库"""
        if not self._check_connection():
            return

        try:
            cursor = self.db_connection.cursor()
            query = """
            INSERT INTO status (device_id, start_timestamp, end_timestamp, status, confidence)
            VALUES (%s, %s, %s, %s, %s)
            """
            # 确保数据类型正确
            values = (
                self.device_id,
                int(start_timestamp),
                int(end_timestamp),
                int(status),
                float(confidence),
            )

            cursor.execute(query, values)
            self.db_connection.commit()
            cursor.close()
            self.logger.info(
                f"已插入状态数据: {start_timestamp} 到 {end_timestamp}, "
                f"status={status}, confidence={confidence:.4f}"
            )
        except Error as e:
            self.logger.error(f"插入状态数据时出错: {e}")

    def start(self):
        """开始预测"""
        self.is_predicting = True
        self.signal_buffer = []
        self.start_timestamp = None
        self.last_prediction_time = 0
        self.receiver.set_handle_csi(self.handle_csi)
        self.receiver.start()
        self.logger.info("开始实时预测...")

    def stop(self):
        """停止预测"""
        if self.is_predicting:
            self.is_predicting = False
            self.receiver.stop()
            self.logger.info("停止预测")

    def close(self):
        """关闭数据库连接"""
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            self.logger.info("数据库连接已关闭")
