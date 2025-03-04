import time
import logging
import json
import numpy as np
import mysql.connector
from mysql.connector import Error
import configparser

class StatusGenerator:
    def __init__(self, device_id='device1'):
        self.device_id = device_id
        self.running = False
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
            self.logger.info("Successfully connected to the database.")
            return connection
        except Error as e:
            self.logger.error(f"Error connecting to MySQL: {e}")
            return None

    def _fetch_csi_data(self, limit=30):
        """
        从数据库中获取未处理的最新 CSI 数据。
        """
        if not self.db_connection:
            return []

        try:
            cursor = self.db_connection.cursor(dictionary=True)
            query = """
            SELECT id, timestamp, csi_data
            FROM raw_data
            WHERE device_id = %s AND processed = 0
            ORDER BY timestamp DESC
            LIMIT %s
            """
            cursor.execute(query, (self.device_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            self.logger.info(f"Fetched {len(rows)} unprocessed CSI records.")
            return rows
        except Error as e:
            self.logger.error(f"Error fetching CSI data: {e}")
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
            WHERE id IN %s
            """
            cursor.execute(query, (tuple(data_ids),))
            self.db_connection.commit()
            cursor.close()
            self.logger.info(f"Marked {len(data_ids)} records as processed.")
        except Error as e:
            self.logger.error(f"Error marking data as processed: {e}")

    def _predict_status(self, csi_data):
        """
        调用机器学习模型进行预测。
        """
        # 这里调用你的机器学习模型
        # 返回预测结果（状态和置信度）
        status = np.random.randint(0, 2)  # 1 表示有人，0 表示无人
        confidence = np.random.uniform(0, 1)  # 置信度
        self.logger.info(f"Predicted status: {status}, confidence: {confidence}")
        return status, confidence

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
            cursor.execute(query, (self.device_id, start_timestamp, end_timestamp, status, confidence))
            self.db_connection.commit()
            cursor.close()
            self.logger.info(f"Inserted status data: {start_timestamp} to {end_timestamp}, status={status}, confidence={confidence}")
        except Error as e:
            self.logger.error(f"Error inserting status data: {e}")

    def _parse_csi_json(self, csi_json):
        """
        将存储的 CSI JSON 数据解析为复数形式的 numpy 数组。
        """
        csi_dict = json.loads(csi_json)
        real_part = np.array(csi_dict['real'], dtype=np.float32)
        imag_part = np.array(csi_dict['imag'], dtype=np.float32)
        return real_part + 1j * imag_part

    def start(self):
        """
        开始生成状态数据。
        """
        self.running = True
        self.logger.info("Start generating status data...")
        while self.running:
            # 获取最新的未处理 CSI 数据
            csi_data = self._fetch_csi_data(limit=30)
            
            # 检查是否恰好有 30 条数据
            if len(csi_data) == 30:
                # 提取时间戳和 CSI 数据
                timestamps = [row['timestamp'] for row in csi_data]
                csi_values = [self._parse_csi_json(row['csi_data']) for row in csi_data]
                data_ids = [row['id'] for row in csi_data]

                # 调用机器学习模型进行预测
                status, confidence = self._predict_status(csi_values)

                # 插入状态数据
                self._insert_status_data(timestamps[0], timestamps[-1], status, confidence)

                # 将处理过的数据标记为已处理
                self._mark_data_as_processed(data_ids)
                
                self.logger.info("Processed 30 CSI records.")
            else:
                # 如果数据不足 30 条，记录日志并等待
                self.logger.info(f"Waiting for more data. Current records: {len(csi_data)}/30")
            
            # 每隔 1 秒检查一次
            time.sleep(1)

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
    status_generator = StatusGenerator(device_id='device1')

    try:
        # 开始生成状态数据
        status_generator.start()
    except KeyboardInterrupt:
        # 用户按下 Ctrl+C 时停止
        status_generator.stop()
    finally:
        # 关闭数据库连接
        status_generator.close()