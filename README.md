# CSI 服务器守护进程

## 使用说明

### 1. 启动守护进程：
```bash
python csi_daemon.py start
python status_daemon.py start
```

### 2. 停止守护进程：
```bash
python csi_daemon.py stop
python status_daemon.py stop
```

### 3. 查看日志：
```bash
tail -f tmp/csi_daemon.log
tail -f tmp/status_daemon.log
```

## 配置文件说明

### 1. 数据库配置文件 (db_config.ini)
```ini
[database]
host = localhost
user = xxx
password = xxx
database = csi
```

### 2. 预测器配置文件 (predictor_config.ini)
```ini
[predictor]
model_name = ResNet50
model_path = ./saved_models/model.pth
signal_process_method = mean_filter
feature_type = 振幅
udp_port = 1234
```

## 功能说明

### 1. CSI守护进程 (csi_daemon.py)
- 使用`predict_realtime_database.py`中的`RealtimePredictorDatabase`类实现实时CSI数据处理和预测
- 接收CSI数据并进行实时预测，每2秒将预测结果（状态和置信度）插入到数据库
- 直接生成状态数据存入数据库，无需后续处理
- 预测结果使用颜色区分不同状态（0：红色，1：绿色）

### 2. 状态守护进程 (status_daemon.py)
- 处理历史CSI数据并更新状态
- 用于批量处理和分析

### 3. 旧版CSI守护进程
- 原先使用`csi_receiver.py`中的`ReadFromUDP`类仅接收CSI数据并存入数据库
- 不包含实时预测功能，只保存原始数据
- 需要由`status_generator.py`统一处理历史数据并生成状态


### 配置文件注意事项
- 所有配置文件应放置在程序运行目录下
- 配置文件使用 INI 格式
- 如果配置文件不存在或读取失败，程序将使用默认配置
- 确保所有配置文件使用 UTF-8 编码