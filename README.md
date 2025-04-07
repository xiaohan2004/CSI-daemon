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
tail -f /tmp/csi_daemon.log
tail -f /tmp/status_daemon.log
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
```

### 配置文件注意事项
- 所有配置文件应放置在程序运行目录下
- 配置文件使用 INI 格式
- 如果配置文件不存在或读取失败，程序将使用默认配置
- 确保所有配置文件使用 UTF-8 编码