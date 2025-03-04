# 使用说明
1. 启动守护进程：
```bash
python csi_daemon.py start
python status_daemon.py start
```

2. 停止守护进程：
```bash
python csi_daemon.py stop
python status_daemon.py stop
```

3. 查看日志：
```bash
tail -f /tmp/csi_daemon.log
tail -f /tmp/status_daemon.log
```

# 配置文件格式
db_config.ini
```ini
[database]
host = localhost
user = xxx
password = xxx
database = csi
```