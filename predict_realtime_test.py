import torch
import numpy as np
import configparser
from receiveCSI import ReadFromUDP, LLTF_VALID_INDEX
from support_web import (
    create_signal2features_preprocess,
    create_model,
    get_last_dim_size,
    create_preprocess_chain,
    single_signal_preprocess_to_matrix_preprocess,
    get_class_by_label,
    find_implementations,
    get_saved_model_path,
)
from abstract import SignalProcessorBase
from models import support
import time


class RealtimePredictor:
    def __init__(
        self,
        model_name,
        model_path,
        signal_process_method,
        feature_type,
        udp_port=1234,
        window_size=support.LEN_W,
    ):
        """
        初始化实时预测器

        参数:
            model_name: 模型名称
            model_path: 已训练模型的文件路径
            signal_process_method: 信号处理方法名称
            feature_type: 特征类型
            udp_port: UDP接收端口
            window_size: 信号窗口大小
        """
        self.window_size = window_size
        self.signal_buffer = []
        self.receiver = ReadFromUDP(udp_port)
        self.is_predicting = False
        self.counter = 0

        # 加载模型
        self.model = create_model(model_name, get_last_dim_size(feature_type, 64))
        self.model.get_model().load_state_dict(
            torch.load(model_path, weights_only=True)
        )
        self.model.get_model().eval()

        # 设置信号处理和特征提取
        signal_processor_methods = find_implementations(SignalProcessorBase)
        denoise_process_cls = get_class_by_label(
            signal_processor_methods, signal_process_method
        )
        denoise_preprocesses = (
            [denoise_process_cls()] if denoise_process_cls is not None else []
        )
        denoise_preprocesses = [
            single_signal_preprocess_to_matrix_preprocess(x)
            for x in denoise_preprocesses
        ]

        # 特征提取
        features_preprocess = create_signal2features_preprocess(feature_type)
        self.preprocess = create_preprocess_chain(
            denoise_preprocesses + [features_preprocess]
        )

    def handle_csi(self, csi_data):
        """处理接收到的CSI数据"""
        if not self.is_predicting:
            return

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
            self.counter += 1
            if self.counter % 10 != 0:
                return
            signal_matrix = np.array(self.signal_buffer)

            # 预处理信号
            processed_signal = self.preprocess(signal_matrix)
            processed_signal = processed_signal.reshape(1, *processed_signal.shape)

            # 转换为tensor并预测
            tensor_signal = torch.tensor(processed_signal).float()
            with torch.no_grad():
                prediction = self.model.get_model()(tensor_signal).numpy()

            # 输出预测结果
            # predicted_class = np.argmax(prediction[0])
            # confidence = prediction[0][predicted_class]
            # print(f"预测类别: {predicted_class}, 置信度: {confidence:.4f}")
            # print(f"完整预测结果: {prediction[0]}")
            # print(f"预测结果: {prediction}")
            predicted_class = np.argmax(prediction[0])
            # 根据类别输出不同的结果，要用颜色区分
            if predicted_class == 0:
                print(
                    "\033[91m预测类别: 0, 置信度: {:.4f}\033[0m".format(
                        prediction[0][0]
                    )
                )  # 红色
            elif predicted_class == 1:
                print(
                    "\033[92m预测类别: 1, 置信度: {:.4f}\033[0m".format(
                        prediction[0][1]
                    )
                )  # 绿色

    def start_predicting(self):
        """开始预测"""
        self.is_predicting = True
        self.signal_buffer = []
        self.receiver.set_handle_csi(self.handle_csi)
        self.receiver.start()
        print("开始实时预测...")

    def stop_predicting(self):
        """停止预测"""
        self.is_predicting = False
        self.receiver.stop()
        print("停止预测")


if __name__ == "__main__":
    # 从配置文件读取配置
    config_parser = configparser.ConfigParser()
    config_parser.read("predictor_config.ini")

    # 配置预测参数
    config = {
        "model_name": config_parser.get("predictor", "model_name"),  # 使用的模型名称
        "model_path": config_parser.get("predictor", "model_path"),  # 模型文件路径
        "signal_process_method": config_parser.get(
            "predictor", "signal_process_method"
        ),  # 信号处理方法
        "feature_type": config_parser.get("predictor", "feature_type"),  # 特征类型
        "udp_port": int(
            config_parser.get("predictor", "udp_port", fallback="1234")
        ),  # UDP接收端口
    }

    print(f"已从配置文件加载设置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 创建预测器实例
    predictor = RealtimePredictor(**config)

    try:
        # 开始预测
        predictor.start_predicting()

        # 持续预测直到用户按下Enter键
        print("按Enter键停止预测...")
        input()

    finally:
        # 停止预测
        predictor.stop_predicting()
