import numpy as np
import torch
import utils
import os

from utils import correct_sampling

LEN_W = 30
STEP_DISTANCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
WIFI_CHANNELS = 64


# DEVICE = torch.device("cpu")

def set_len_w(len_w):
    global LEN_W
    LEN_W = len_w

def set_step_distance(step_distance):
    global STEP_DISTANCE
    STEP_DISTANCE = step_distance

def set_batch_size(batch_size):
    global BATCH_SIZE
    BATCH_SIZE = batch_size


def set_device(device):
    global DEVICE
    DEVICE = device


def get_default_save(path):
    """
    获取默认保存函数
    :param path: 保持模型参数的文件路径
    :return: 函数接受两个参数，一个是模型，一个是loss，保存历史loss最小的模型参数
    """
    loss_min = None

    def save(model, loss):
        nonlocal loss_min
        if loss_min is None or loss_min > loss:
            loss_min = loss
            print("min_loss", loss_min)
            torch.save(model.state_dict(), path)

    return save


def load_model(model, path):
    """
    加载模型
    :param model: 要加载参数的模型例
    :param path: 模型参数文件路径
    :return: 加载后的模型
    """
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def try_load_model(model, path):
    """
    尝试加载模型，如果文件不存在则不加载
    :param model: 要加载参数的模型例
    :param path: 模型参数文件路径
    :return: 加载后的模型或者原模型
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    return model


# 将列表中元素比上一个元素小的元素补上k的倍数，令其恰好比上一个元素大
def make_up_list(data, k):
    for i in range(1, len(data)):
        if data[i] <= data[i - 1]:
            data[i] = data[i - 1] + k
    return data


def get_dataloader(domain=0, split=True, step_distance=STEP_DISTANCE, preprocess=None, batch_size=None,
                   data_path_prefix=".."):
    import scipy
    import numpy as np
    from torch.utils.data import TensorDataset, random_split, DataLoader
    global LEN_W

    def get_amplitude_and_phase(file):
        packets = utils.read_udp_data_txt_to_bytes(file)
        s = [utils.CSI.get_csi_vector_from_packet(x) for x in packets]
        s_time = [utils.CSI.get_tsf_from_pack(x) for x in packets]
        s = np.where(np.isnan(s), 0, s)
        # s = np.concatenate((s[:, 1:27], s[:, 38:64]), axis=1)
        # global WIFI_CHANNELS
        # WIFI_CHANNELS = 52
        # return np.log10(np.abs(s) + 1e-10), s_time
        # return np.stack((np.log10(np.abs(s) + 1e-10), np.angle(s)), axis=-1), s_time
        return s, s_time

    if domain == 0:
        m2 = get_amplitude_and_phase(f"{data_path_prefix}/0510/2M.txt")
        m4 = get_amplitude_and_phase(f"{data_path_prefix}/0510/4M.txt")
        me = get_amplitude_and_phase(f"{data_path_prefix}/0510/metal.pcap.txt")
        me1 = get_amplitude_and_phase(f"{data_path_prefix}/0510/metal1.pcap.txt")

        need_for = [(m2, 0), (m4, 1), (me, 2), (me1, 3)]
    elif domain == 1:
        my_m2 = get_amplitude_and_phase(f"{data_path_prefix}/data/my_2m.txt")
        my_m4 = get_amplitude_and_phase(f"{data_path_prefix}/data/my_4m.txt")

        need_for = [(my_m2, 0), (my_m4, 1)]
    elif domain == 2:
        workroom_has_people = get_amplitude_and_phase(
            f"{data_path_prefix}/0510/2024-10-06-23-40.workroom.has_people.txt")
        workroom_has_people = (workroom_has_people[0][1000:2000], workroom_has_people[1][1000:2000])
        workroom_no_people = get_amplitude_and_phase(f"{data_path_prefix}/0510/2024-10-07-01-19.workroom.no_people.txt")
        need_for = [(workroom_has_people, 0), (workroom_no_people, 1)]
    elif domain == 3:
        restroom_has_people = get_amplitude_and_phase(
            f"{data_path_prefix}/data/2024-10-07-14_23.restroom.10-31.30minutes.has_people.txt")
        restroom_has_people = (restroom_has_people[0][:5000], restroom_has_people[1][:5000])
        restroom_no_people = get_amplitude_and_phase(
            f"{data_path_prefix}/data/2024-10-07-15_27.restroom.10-31.10minutes.no_people.txt")
        restroom_no_people = (restroom_no_people[0][:5000], restroom_no_people[1][:5000])
        need_for = [(restroom_has_people, 0), (restroom_no_people, 1)]
    elif domain == -1:
        m2 = get_amplitude_and_phase(f"{data_path_prefix}/0510/2M.txt")
        m4 = get_amplitude_and_phase(f"{data_path_prefix}/0510/4M.txt")
        me = get_amplitude_and_phase(f"{data_path_prefix}/0510/metal.pcap.txt")
        me1 = get_amplitude_and_phase(f"{data_path_prefix}/0510/metal1.pcap.txt")

        my_m2 = get_amplitude_and_phase(f"{data_path_prefix}/data/my_2m.txt")
        my_m4 = get_amplitude_and_phase(f"{data_path_prefix}/data/my_4m.txt")
        need_for = [(m2, 0), (m4, 1), (me, 2), (me1, 3), (my_m2[:200], 0), (my_m4[:200], 1)]
    else:
        raise ValueError("domain must be 0，1，2,3")
    data = []
    labels = []

    for (m, label) in need_for:
        max_k = 9
        ############################
        # 尽管get_dataloader_from_csv 由于没有时间戳，所以不能时间对齐。
        # 但是这里既然有了时间戳，就进行一下优化吧。
        ############################
        # 时间戳对齐
        print("m.shape:", m)
        for i in range(0, len(m[0]) - max_k * LEN_W, step_distance):
            time = np.zeros(LEN_W)
            k = 0
            while k == 0 or time[-1] - time[0] < 16 * LEN_W:
                k += 1
                matrix = m[0][i:i + k * LEN_W]
                time = m[1][i:i + k * LEN_W]
                time = np.array(time)
                time = (time / 1.0e6)  # 降低时间精度，减少数据量
                time = make_up_list(time, 2 ** 32 / 1e6)

            matrix = np.apply_along_axis(correct_sampling, 0, matrix, time, target_interval=16, return_size=LEN_W)
            data.append(matrix)
            labels.append(label)

    # 为了和get_dataloader_from_csv 对齐，这里去除了对数据的处理############
    # # decrease noise
    # data = [np.apply_along_axis(scipy.signal.medfilt, 1, x.real, kernel_size=5) +
    #         np.apply_along_axis(scipy.signal.medfilt, 1, x.imag, kernel_size=5) * 1j for x in data]

    if preprocess is not None:
        data = [preprocess(x) for x in data]
    # print(f"data:{data}")
    # _ = input("pause")
    data = torch.tensor(data, dtype=torch.float32)
    print(data.shape)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data, labels)

    if split:
        # Split the dataset into training and testing sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 创建 DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE if batch_size is None else batch_size,
                                      shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE if batch_size is None else batch_size,
                                     shuffle=True)

        return train_dataloader, test_dataloader
    else:
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def catch_cancel(fun):
    try:
        fun()
    except KeyboardInterrupt:
        print("catch KeyboardInterrupt")


def simple_train(model, train_loader, valid_loader, epochs=1000, save=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if not save is None:
                save(loss)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Epoch:{}, Loss_sum:{:.4f}'.format(epoch + 1, total_loss))
        if total_loss < 1e-10:
            break

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the valid data: %d %%' % (100 * correct / total))


def l2_regex_train(model, lambda_arg, train_loader, valid_loader, epochs=1000, save=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    l2_lambda = lambda_arg
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 正则化避免过拟合
            l2_reg = torch.tensor(0., requires_grad=True).to(DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            if not save is None:
                save(loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Epoch:{}, Loss_sum:{:.4f}'.format(epoch + 1, total_loss))
        if total_loss < 1e-10:
            break

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the valid data: %d %%' % (100 * correct / total))


def test(model, dataloader):
    from sklearn.metrics import confusion_matrix
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    get_dataloader(3)
