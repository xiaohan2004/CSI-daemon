## 用于web页面的支持函数以及全局变量
import inspect
import os
import typing
import numpy as np
import torch
import models.LP_RNN
import utils
from abstract import SignalReaderBase
from models import support
from models.All import Model
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 常量定义
SAVED_MODELS_PATH = "./saved_models"
SAVED_CSV_PATH = "./saved_csv"
SUPPORTED_MODELS = [
    "FC",
    "RNN",
    "GRU",
    "LSTM",
    "BiLSTM",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "LP_RNN",
]
SUPPORTED_FEATURES = [
    "CSI功率",
    "振幅",
    "CSI相位",
    "DFS",
    "CSI功率+振幅",
    "CSI功率+相位",
]
N_CLASS = 2
SUPPORTED_DATASET = ["环境1", "环境2", "自定义"]


def get_saved_model_path(file_name):
    if not os.path.exists(SAVED_MODELS_PATH):
        os.makedirs(SAVED_MODELS_PATH)
    return f"{SAVED_MODELS_PATH}/{support.LEN_W}-{support.STEP_DISTANCE}-{file_name}"


def get_saved_csv_path(file_name):
    if not os.path.exists(SAVED_CSV_PATH):
        os.makedirs(SAVED_CSV_PATH)
    return f"{SAVED_CSV_PATH}/{file_name}"


def get_producer(cls, *args, **kwargs):
    if not hasattr(get_producer, "RUNNING_MAP"):
        get_producer.RUNNING_MAP = {}

    if (
        cls in get_producer.RUNNING_MAP
        and not get_producer.RUNNING_MAP[cls].is_stopped()
    ):
        get_producer.RUNNING_MAP[cls].stop()
        while not get_producer.RUNNING_MAP[cls].is_stopped():
            print("waiting for producer to stop", cls)
            pass

    get_producer.RUNNING_MAP[cls] = cls(*args, **kwargs)
    return get_producer.RUNNING_MAP[cls]


def get_class_by_label(classes, label):
    for cls in classes:
        if cls.get_label() == label:
            return cls
    return None


def single_signal_preprocess_to_matrix_preprocess(preprocess):
    def multi_signal_preprocess(signals, *args, **kwargs):
        return np.apply_along_axis(preprocess, 0, signals, *args, **kwargs)

    return multi_signal_preprocess


def create_signal2features_preprocess(features):
    def small_wave_preprocess(signal):
        import pywt

        fs = 100  # 采样频率
        care_fs = 2  # 关心的频率上限
        base_len = 40  # 返回的长度
        wavename = "cmor3-3"
        totalscal = base_len * fs / care_fs
        Fc = pywt.central_frequency(wavename)
        c = 2 * Fc * totalscal
        scales = c / np.arange(1, totalscal + 1)
        coefs, _ = pywt.cwt(signal, scales, wavename, 1 / fs)
        return coefs[:base_len].T

    def small_wave_preprocess_for_m(m):
        m = [small_wave_preprocess(m[:, i]) for i in range(m.shape[1])]
        ret = np.zeros((m[0].shape[0], m[0].shape[1] * len(m)))
        for i in range(len(m)):
            ret[:, i * m[0].shape[1] : (i + 1) * m[0].shape[1]] = m[i]
        return ret

    funcs = {
        "CSI功率": lambda x: utils.CSI.get_amplitude_db_unit(x),
        "振幅": lambda x: utils.CSI.get_amplitude(x),
        "DFS": lambda x: small_wave_preprocess_for_m(utils.CSI.get_amplitude(x)),
        "CSI相位": lambda x: utils.CSI.get_phase(x),
        "CSI功率+振幅": lambda x: np.concatenate(
            [utils.CSI.get_amplitude_db_unit(x), utils.CSI.get_amplitude(x)], axis=1
        ),
        "CSI功率+相位": lambda x: np.concatenate(
            [utils.CSI.get_amplitude_db_unit(x), utils.CSI.get_phase(x)], axis=1
        ),
    }
    if features in funcs:
        return funcs[features]
    elif features in SUPPORTED_FEATURES:
        raise NotImplementedError("Not implemented for :" + features)
    else:
        raise ValueError("Unknown features")


def get_last_dim_size(features, common_size):
    if features in ["CSI功率", "振幅", "CSI相位"]:
        return common_size * 1
    elif features in ["CSI功率+振幅", "CSI功率+相位"]:
        return common_size * 2
    elif features in ["DFS"]:
        return common_size * 20
    elif features in SUPPORTED_FEATURES:
        raise NotImplementedError("Not implemented for :" + features)
    else:
        raise ValueError("Unknown features")


def create_model(model_name, last_dim_size) -> Model:
    global N_CLASS
    n_classes = N_CLASS
    from models.All import (
        SimpleMLP,
        SimpleLSTM,
        SimpleBiLSTM,
        SimpleGRU,
        SimpleRNN,
        SimpleResNet18,
        SimpleResNet34,
        SimpleResNet50,
    )

    models_map = {
        "FC": lambda: SimpleMLP(last_dim_size=last_dim_size, num_classes=n_classes),
        "RNN": lambda: SimpleRNN(last_dim_size=last_dim_size, num_classes=n_classes),
        "GRU": lambda: SimpleGRU(last_dim_size=last_dim_size, num_classes=n_classes),
        "LSTM": lambda: SimpleLSTM(last_dim_size, n_classes, 64),
        "BiLSTM": lambda: SimpleBiLSTM(last_dim_size, n_classes, 64),
        "ResNet18": lambda: SimpleResNet18(support.LEN_W),
        "ResNet34": lambda: SimpleResNet34(support.LEN_W),
        "ResNet50": lambda: SimpleResNet50(support.LEN_W),
        "LP_RNN": lambda: models.LP_RNN.SimpleLP_RNN(
            last_dim_size=last_dim_size, num_classes=n_classes
        ),
    }

    if model_name in models_map:
        return models_map[model_name]()
    elif model_name in SUPPORTED_MODELS:
        raise NotImplementedError(f"Not implemented for: {model_name}")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_dataloader(domain, preprocess, split=True, csv_files_with_labels=None):
    domain_map = {
        "环境1": 2,
        "环境2": 3,
    }

    if domain in domain_map:
        return support.get_dataloader(
            domain=domain_map[domain],
            split=split,
            preprocess=preprocess,
            data_path_prefix=".",
        )
    elif domain == "自定义":
        return get_dataloader_from_csv(
            csv_files_with_labels, split=split, preprocess=preprocess
        )
    else:
        raise ValueError(f"Unknown domain: {domain}")


# def get_dataloader_from_csv(
#     csv_files_with_labels,
#     split=True,
#     step_distance=support.STEP_DISTANCE,
#     preprocess=None,
#     batch_size=support.BATCH_SIZE,
#     data_path_prefix=".",
# ):
#     import numpy as np
#     from torch.utils.data import TensorDataset, random_split, DataLoader

#     LEN_W = support.LEN_W
#     raw_data = (
#         [
#             np.loadtxt(
#                 f"{data_path_prefix}/{csv_file}", delimiter=",", dtype=np.complex64
#             )
#             for csv_file in csv_files_with_labels[0]
#         ],
#         csv_files_with_labels[1],
#     )

#     data = []
#     labels = []

#     for m, label in zip(raw_data[0], raw_data[1]):
#         for i in range(0, m.shape[0] - LEN_W, step_distance):
#             matrix = m[i : i + LEN_W :, :]
#             data.append(matrix)
#             labels.append(int(label))

#     if preprocess is not None:
#         data = [preprocess(x) for x in data]

#     data = np.array(data)
#     data = torch.tensor(data, dtype=torch.float32)
#     labels = torch.tensor(labels, dtype=torch.long)
#     dataset = TensorDataset(data, labels)

#     if split:
#         train_size = int(0.8 * len(dataset))
#         test_size = len(dataset) - train_size
#         train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#         train_dataloader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True
#         )
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#         return train_dataloader, test_dataloader
#     else:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=True)
def get_dataloader_from_csv(
    csv_files_with_labels,
    split=True,
    step_distance=support.STEP_DISTANCE,
    preprocess=None,
    batch_size=support.BATCH_SIZE,
    data_path_prefix=".",
):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, random_split, DataLoader
    import time
    import psutil
    import os
    from tqdm import tqdm
    
    def print_debug_info(step_name, start_time, prev_memory):
        current_time = time.time()
        elapsed = current_time - start_time
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_diff = current_memory - prev_memory
        print(f"[DEBUG] {step_name}: Time={elapsed:.3f}s, Memory={current_memory:.2f}MB (+{memory_diff:.2f}MB)")
        return current_time, current_memory
    
    total_start = time.time()
    prev_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print("\n[DEBUG] Starting get_dataloader_from_csv...")
    
    LEN_W = support.LEN_W
    
    # 1. Loading CSV files
    print("\n[DEBUG] Step 1: Loading CSV files...")
    load_start = time.time()
    raw_data_files = []
    # 添加进度条
    for csv_file in tqdm(csv_files_with_labels[0], desc="Loading CSV files"):
        raw_data_files.append(
            np.loadtxt(
                f"{data_path_prefix}/{csv_file}", delimiter=",", dtype=np.complex64
            )
        )
    raw_data = (raw_data_files, csv_files_with_labels[1])
    _, prev_memory = print_debug_info("Loaded CSV files", load_start, prev_memory)
    
    # 2. Processing data into windows
    print("\n[DEBUG] Step 2: Processing data into windows...")
    process_start = time.time()
    data = []
    labels = []

    # 添加两层进度条
    pbar_files = tqdm(zip(raw_data[0], raw_data[1]), total=len(raw_data[0]), desc="Processing files")
    for m, label in pbar_files:
        # 更新外部进度条描述
        pbar_files.set_description(f"Processing file (shape={m.shape})")
        
        # 内部进度条
        total_windows = len(range(0, m.shape[0] - LEN_W, step_distance))
        pbar_windows = tqdm(range(0, m.shape[0] - LEN_W, step_distance), 
                          total=total_windows, 
                          desc="Creating windows", 
                          leave=False)
        
        for i in pbar_windows:
            matrix = m[i : i + LEN_W :, :]
            data.append(matrix)
            labels.append(int(label))
        pbar_windows.close()
    
    print(f"[DEBUG] Generated {len(data)} windows from {len(raw_data[0])} files")
    _, prev_memory = print_debug_info("Processed data windows", process_start, prev_memory)
    
    # 3. Preprocessing if needed
    if preprocess is not None:
        print("\n[DEBUG] Step 3: Applying preprocessing...")
        preprocess_start = time.time()
        # 添加预处理进度条
        data = [preprocess(x) for x in tqdm(data, desc="Preprocessing data")]
        _, prev_memory = print_debug_info("Applied preprocessing", preprocess_start, prev_memory)
    
    # 4. Converting to tensors
    print("\n[DEBUG] Step 4: Converting to tensors...")
    tensor_start = time.time()
    data = np.array(data)
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(data, labels)
    _, prev_memory = print_debug_info("Converted to tensors", tensor_start, prev_memory)
    
    # 5. Splitting dataset
    print("\n[DEBUG] Step 5: Splitting dataset...")
    split_start = time.time()
    if split:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        _, prev_memory = print_debug_info("Split dataset", split_start, prev_memory)
        
        total_elapsed = time.time() - total_start
        print(f"\n[DEBUG] Total execution time: {total_elapsed:.3f}s")
        return train_dataloader, test_dataloader
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        _, prev_memory = print_debug_info("Created dataloader", split_start, prev_memory)
        
        total_elapsed = time.time() - total_start
        print(f"\n[DEBUG] Total execution time: {total_elapsed:.3f}s")
        return dataloader


class SimpleSignalReader(SignalReaderBase):
    def __init__(self, receiver):
        self.receiver = receiver

    def read(self, signal: np.ndarray, *args: typing.Any, **kwargs: typing.Any):
        self.receiver(signal, *args, **kwargs)


def find_implementations(abstraction: typing.Type) -> [typing.Type]:
    implementations = []
    for subclass in abstraction.__subclasses__():
        if not inspect.isabstract(subclass):
            implementations.append(subclass)
        implementations.extend(find_implementations(subclass))
    return implementations


def create_preprocess_chain(processors):
    def preprocess_chain(signal):
        for p in processors:
            if p is not None:
                signal = p(signal)
        return signal

    return preprocess_chain


if __name__ == "__main__":
    import signalProcessorInstance

    get_dataloader(
        "自定义",
        lambda x: x,
        csv_files_with_labels=(
            ["./saved_csv/read_from_serial_2024-11-11_10-42-52.csv"],
            [1],
        ),
    )
