-- 1. 检查并删除已存在的数据库
DROP DATABASE IF EXISTS csi;

-- 2. 创建数据库
CREATE DATABASE csi;

-- 3. 使用数据库
USE csi;

-- 4. 创建原始数据表
CREATE TABLE raw_data (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,  -- 自增主键
    device_id VARCHAR(32) NOT NULL,        -- 设备ID
    timestamp BIGINT NOT NULL,        -- 校正后的时间戳（精确到毫秒）
    csi_data JSON NOT NULL,                -- CSI数据（JSON格式）
    INDEX idx_device_timestamp (device_id, timestamp)  -- 设备和时间戳的联合索引
);

-- 5. 创建状态数据表
CREATE TABLE status (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,  -- 自增主键
    device_id VARCHAR(32) NOT NULL,        -- 设备ID
    start_timestamp BIGINT NOT NULL, -- 状态开始时间
    end_timestamp BIGINT NOT NULL,   -- 状态结束时间
    status TINYINT(1) NOT NULL,           -- 预测结果（1: 有人, 0: 无人）
    confidence FLOAT NOT NULL,            -- 预测置信度
    INDEX idx_device_end_time (device_id, end_timestamp DESC)  -- 设备和结束时间的联合索引
);