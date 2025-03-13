import os
import sys
import time
import signal
import logging
from status_generator import StatusGenerator


def setup_logger(log_file):
    """
    配置日志记录器，将日志输出到文件。
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建一个格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 清除所有已存在的处理器
    root_logger.handlers = []

    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


def daemonize(pid_file):
    """
    将当前进程转换为守护进程。
    """
    # 第一次 fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)  # 退出父进程

    # 创建新的会话
    os.setsid()

    # 第二次 fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)  # 退出父进程

    # 写入 PID 文件
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    # 重定向标准输入、输出和错误
    sys.stdout.flush()
    sys.stderr.flush()
    with open("/dev/null", "r") as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open("/dev/null", "a") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())


def start_daemon(pid_file, log_file):
    """
    启动守护进程。
    """
    if os.path.exists(pid_file):
        print("Status generator daemon is already running.")
        sys.exit(1)

    # 配置日志
    logger = setup_logger(log_file)
    logger.info("Starting status generator daemon...")

    # 转换为守护进程
    daemonize(pid_file)

    # 守护进程的主逻辑
    generator = StatusGenerator(device_id="device1")
    generator.start()

    try:
        while True:
            time.sleep(1)  # 保持守护进程运行
    except KeyboardInterrupt:
        logger.info("Stopping status generator daemon...")
        generator.stop()


def stop_daemon(pid_file):
    """
    停止守护进程。
    """
    if not os.path.exists(pid_file):
        print("Status generator daemon is not running.")
        return

    with open(pid_file, "r") as f:
        pid = int(f.read())

    try:
        os.kill(pid, signal.SIGTERM)  # 发送终止信号
        os.remove(pid_file)  # 删除 PID 文件
        print("Status generator daemon stopped.")
    except ProcessLookupError:
        print("Status generator daemon process not found.")
        os.remove(pid_file)


if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义 tmp 目录路径
    tmp_dir = os.path.join(current_dir, "tmp")

    # 确保 tmp 目录存在
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # 定义 PID 文件和日志文件路径
    pid_file = os.path.join(tmp_dir, "status_daemon.pid")  # PID 文件路径
    log_file = os.path.join(tmp_dir, "status_daemon.log")  # 日志文件路径

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} start|stop")
        sys.exit(1)

    if sys.argv[1] == "start":
        start_daemon(pid_file, log_file)
    elif sys.argv[1] == "stop":
        stop_daemon(pid_file)
    else:
        print(f"Usage: {sys.argv[0]} start|stop")
        sys.exit(1)
