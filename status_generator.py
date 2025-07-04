import time
import logging
import json
import numpy as np
import mysql.connector
from mysql.connector import Error
import configparser
from predict import Predictor

LEN = 200  # 每个 CSI 数据的长度


class StatusGenerator:
    def __init__(self, device_id="device1"):
        self.device_id = device_id
        self.running = False
        # 使用模块名作为 logger 名称
        self.logger = logging.getLogger(__name__)
        self.db_connection = self._connect_to_database()

        # 从配置文件初始化预测器
        predictor_config = self._load_predictor_config()
        self.predictor = Predictor(**predictor_config)

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
                # 设置连接超时
                connection_timeout=30,
                # 禁用查询缓存
                consume_results=True,
                # 添加连接参数
                sql_mode="STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION",
            )

            # 设置会话变量
            cursor = connection.cursor()
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
            cursor.execute("SET SESSION innodb_lock_wait_timeout=50")
            cursor.close()

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

    def _fetch_csi_data(self, limit=LEN):
        """
        从数据库中获取未处理的最新 CSI 数据。
        """
        # 检查并确保数据库连接
        if not self._check_connection():
            return []

        try:
            # 开启新事务
            self.db_connection.commit()

            cursor = self.db_connection.cursor(dictionary=True)

            # 设置事务隔离级别为 READ COMMITTED
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")

            # 添加更多调试信息
            verify_query = """
            SELECT COUNT(*) as total 
            FROM raw_data 
            WHERE device_id = %s AND processed = 0
            """
            cursor.execute(verify_query, (self.device_id,))
            total = cursor.fetchone()["total"]
            self.logger.debug(f"数据库中共有 {total} 条未处理记录")

            query = """
            SELECT /*+ NO_CACHE */ id, timestamp, csi_data
            FROM raw_data
            WHERE device_id = %s 
            AND processed = 0
            ORDER BY timestamp
            LIMIT %s
            """
            self.logger.debug(
                f"执行查询: {query} 参数: device_id={self.device_id}, limit={limit}"
            )

            cursor.execute(query, (self.device_id, limit))
            rows = cursor.fetchall()

            # 确保提交事务
            self.db_connection.commit()
            cursor.close()

            # 添加更详细的日志
            if not rows:
                self.logger.warning(
                    f"未找到未处理的数据记录 (device_id={self.device_id})"
                )
            else:
                self.logger.info(f"获取到 {len(rows)} 条未处理的记录")
                self.logger.debug(f"第一条记录 ID: {rows[0]['id'] if rows else 'N/A'}")
                self.logger.debug(
                    f"最后一条记录 ID: {rows[-1]['id'] if rows else 'N/A'}"
                )
                # 添加时间戳范围信息
                self.logger.debug(
                    f"时间戳范围: {rows[-1]['timestamp']} 到 {rows[0]['timestamp']}"
                )

            return rows
        except Error as e:
            self.logger.error(f"获取 CSI 数据时发生错误: {e}")
            # 发生错误时关闭连接，下次查询时会重新连接
            self.close()
            return []

    def _mark_data_as_processed(self, data_ids):
        """
        将指定 ID 的数据标记为已处理。
        """
        if not self.db_connection or not data_ids:
            return

        try:
            cursor = self.db_connection.cursor()
            query = """
            UPDATE raw_data
            SET processed = 1
            WHERE id IN (%s)
            """ % ",".join(
                ["%s"] * len(data_ids)
            )
            cursor.execute(query, tuple(data_ids))
            self.db_connection.commit()
            cursor.close()
            self.logger.info(f"Marked {len(data_ids)} records as processed.")
        except Error as e:
            self.logger.error(f"Error marking data as processed: {e}")

    def _predict_status(self, csi_values):
        """
        使用预测器进行预测。

        参数:
            csi_values: 包含LEN个CSI数据的列表，每个元素是一个复数numpy数组

        返回:
            (status, confidence): 预测的状态(0或1)和置信度
        """
        try:
            # 将CSI数据转换为numpy数组
            csi_data = np.array(csi_values)

            # 调用预测器进行预测
            predictions = self.predictor.predict_from_array(csi_data)

            if predictions:
                # 获取最后一个预测结果
                last_prediction = predictions[-1]
                status = last_prediction["class"]
                confidence = float(last_prediction["confidence"])

                self.logger.info(f"预测状态: {status}, 置信度: {confidence:.4f}")
                return status, confidence
            else:
                self.logger.warning("预测失败，返回默认值")
                return 0, 0.0

        except Exception as e:
            self.logger.error(f"预测过程中发生错误: {e}")
            return 0, 0.0

    def _insert_status_data(self, start_timestamp, end_timestamp, status, confidence):
        """
        将状态数据插入数据库。
        """
        if not self.db_connection:
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
                int(start_timestamp),  # 转换为普通的 Python int
                int(end_timestamp),  # 转换为普通的 Python int
                int(status),  # 转换为普通的 Python int
                float(confidence),  # 转换为普通的 Python float
            )

            cursor.execute(query, values)
            self.db_connection.commit()
            cursor.close()
            self.logger.info(
                f"Inserted status data: {start_timestamp} to {end_timestamp}, "
                f"status={status}, confidence={confidence}"
            )
        except Error as e:
            self.logger.error(f"Error inserting status data: {e}")

    def _parse_csi_json(self, csi_json):
        """
        将存储的 CSI JSON 数据解析为复数形式的 numpy 数组。
        """
        csi_dict = json.loads(csi_json)
        real_part = np.array(csi_dict["real"], dtype=np.float32)
        imag_part = np.array(csi_dict["imag"], dtype=np.float32)
        return real_part + 1j * imag_part

    def _load_predictor_config(self):
        """
        从配置文件加载预测器配置。
        """
        try:
            config = configparser.ConfigParser()
            config.read("predictor_config.ini")

            predictor_config = {
                "model_name": config.get("predictor", "model_name"),
                "model_path": config.get("predictor", "model_path"),
                "signal_process_method": config.get(
                    "predictor", "signal_process_method"
                ),
                "feature_type": config.get("predictor", "feature_type"),
            }

            self.logger.info(f"成功加载预测器配置: {predictor_config}")
            return predictor_config
        except Exception as e:
            self.logger.error(f"加载预测器配置失败: {e}")
            # 使用默认配置
            default_config = {
                "model_name": "ResNet50",
                "model_path": "./saved_models/model.pth",
                "signal_process_method": "mean_filter",
                "feature_type": "振幅",
            }
            self.logger.warning(f"使用默认配置: {default_config}")
            return default_config

    def start(self):
        """
        开始生成状态数据。
        """
        self.running = True
        self.logger.info(f"开始生成状态数据... (device_id={self.device_id})")

        last_success_time = time.time()
        consecutive_empty_count = 0

        while self.running:
            # 获取最新的未处理 CSI 数据
            self.logger.debug("尝试获取新的 CSI 数据...")
            csi_data = self._fetch_csi_data(limit=LEN * 21)

            current_time = time.time()

            # 检查是否有足够的数据
            if len(csi_data) == LEN * 21:
                consecutive_empty_count = 0
                last_success_time = current_time

                # 提取时间戳和 CSI 数据
                timestamps = [row["timestamp"] for row in csi_data]
                csi_values = [self._parse_csi_json(row["csi_data"]) for row in csi_data]
                data_ids = [row["id"] for row in csi_data]

                csi_values = csi_values[LEN * 10 : LEN * 11]
                # 调用机器学习模型进行预测
                status, confidence = self._predict_status(csi_values)

                # 插入状态数据，使用LEN条数据的时间范围
                self._insert_status_data(
                    timestamps[0], timestamps[-1], status, confidence
                )

                # 将所有LEN条数据标记为已处理
                self._mark_data_as_processed(data_ids)

                self.logger.info(f"Processed {LEN} CSI records.")
            else:
                # 添加更详细的等待信息
                consecutive_empty_count += 1
                time_since_last_success = current_time - last_success_time

                self.logger.info(
                    f"等待更多数据。当前记录数: {len(csi_data)}/{LEN}, "
                    f"已等待: {time_since_last_success:.1f}秒, "
                    f"连续等待次数: {consecutive_empty_count}"
                )

                if len(csi_data) > 0:
                    self.logger.debug(f"最新记录时间戳: {csi_data[0]['timestamp']}")

                # 如果长时间没有新数据，可能需要检查数据源
                if consecutive_empty_count > 20:  # 大约30秒
                    self.logger.warning(
                        f"已经 {time_since_last_success:.1f} 秒没有收到足够的新数据，"
                        f"请检查数据源是否正常工作"
                    )

    def stop(self):
        """
        停止生成状态数据。
        """
        self.running = False
        self.logger.info("Status generator stopped.")

    def close(self):
        """
        关闭数据库连接。
        """
        if self.db_connection:
            self.db_connection.close()
            self.logger.info("Database connection closed.")


# 示例使用
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建 StatusGenerator 实例
    status_generator = StatusGenerator(device_id="device1")

    try:
        # 开始生成状态数据
        status_generator.start()
    except KeyboardInterrupt:
        # 用户按下 Ctrl+C 时停止
        status_generator.stop()
    finally:
        # 关闭数据库连接
        status_generator.close()
