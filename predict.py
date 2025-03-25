import torch
import numpy as np
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
from receiveCSI import LLTF_VALID_INDEX


class Predictor:
    def __init__(
        self,
        model_name,
        model_path,
        signal_process_method,
        feature_type,
        window_size=support.LEN_W,
    ):
        """
        初始化文件预测器

        参数:
            model_name: 模型名称
            model_path: 已训练模型的文件路径
            signal_process_method: 信号处理方法名称
            feature_type: 特征类型
            window_size: 信号窗口大小
        """
        self.window_size = window_size

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

    def read_csi_from_file(self, file_path):
        """从CSV文件读取CSI数据"""
        csi_data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    # 解析复数字符串，格式如："1.23+4.56j"
                    complex_numbers = [complex(x) for x in line.split(",")]
                    csi_data.append(complex_numbers)
        return np.array(csi_data)

    def predict(self, file_path):
        """预测指定文件中的CSI数据"""
        # 读取数据
        csi_data = self.read_csi_from_file(file_path)

        # 处理无效子载波
        for i in range(len(csi_data)):
            invalid_index = [j for j in range(0, 64) if j not in LLTF_VALID_INDEX]
            csi_data[i][invalid_index] = 0

        # 如果数据量不足window_size，则无法预测
        if len(csi_data) < self.window_size:
            print(
                f"数据量不足，需要至少 {self.window_size} 组数据，当前只有 {len(csi_data)} 组"
            )
            return None

        predictions = []
        # 使用滑动窗口进行预测
        for i in range(0, len(csi_data) - self.window_size + 1):
            window_data = csi_data[i : i + self.window_size]

            # 预处理信号
            processed_signal = self.preprocess(window_data)
            processed_signal = processed_signal.reshape(1, *processed_signal.shape)

            # 转换为tensor并预测
            tensor_signal = torch.tensor(processed_signal).float()
            with torch.no_grad():
                prediction = self.model.get_model()(tensor_signal).numpy()

            # 记录预测结果
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            predictions.append(
                {
                    "class": predicted_class,
                    "confidence": confidence,
                    "full_prediction": prediction[0],
                }
            )

        return predictions

    def predict_from_array(self, csi_data):
        """
        直接从CSI数组进行预测

        参数:
            csi_data: numpy数组，形状为(n, 64)的CSI数据，每行是一个时间点的64个子载波数据

        返回:
            predictions: 预测结果列表
        """
        # 处理无效子载波
        for i in range(len(csi_data)):
            invalid_index = [j for j in range(0, 64) if j not in LLTF_VALID_INDEX]
            csi_data[i][invalid_index] = 0

        # 如果数据量不足window_size，则无法预测
        if len(csi_data) < self.window_size:
            print(
                f"数据量不足，需要至少 {self.window_size} 组数据，当前只有 {len(csi_data)} 组"
            )
            return None

        predictions = []
        # 使用滑动窗口进行预测
        for i in range(0, len(csi_data) - self.window_size + 1):
            window_data = csi_data[i : i + self.window_size]

            # 预处理信号
            processed_signal = self.preprocess(window_data)
            processed_signal = processed_signal.reshape(1, *processed_signal.shape)

            # 转换为tensor并预测
            tensor_signal = torch.tensor(processed_signal).float()
            with torch.no_grad():
                prediction = self.model.get_model()(tensor_signal).numpy()

            # 记录预测结果
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            predictions.append(
                {
                    "class": predicted_class,
                    "confidence": float(confidence),  # 确保可以被JSON序列化
                    "full_prediction": prediction[
                        0
                    ].tolist(),  # 转换为list以便JSON序列化
                }
            )

        return predictions


if __name__ == "__main__":
    # 配置预测参数
    config = {
            "model_name": "ResNet50",  # 使用的模型名称
            "model_path": "model.pth",  # 模型文件路径
            "signal_process_method": "mean_filter",  # 信号处理方法
            "feature_type": "振幅",  # 特征类型
    }

    # 创建预测器实例
    predictor = Predictor(**config)

    file_path = "./predict.csv"

    try:
        # 进行预测
        predictions = predictor.predict(file_path)
        if predictions:
            print("\n预测结果:")
            # 统计各类别的预测次数
            class_counts = {}
            for p in predictions:
                class_counts[p["class"]] = class_counts.get(p["class"], 0) + 1

            # 输出最终预测结果（投票结果）
            most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
            print(f"\n最终预测类别（投票结果）: {most_common_class}")
            print(f"各类别预测次数: {class_counts}")

            # 输出详细预测结果
            print("\n详细预测结果:")
            for i, p in enumerate(predictions):
                print(f"窗口 {i+1}: 类别 {p['class']}, 置信度 {p['confidence']:.4f}")
                print(f"完整预测结果: {p['full_prediction']}")

    except Exception as e:
        print(f"错误: {str(e)}")
