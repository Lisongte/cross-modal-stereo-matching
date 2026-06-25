"""
数据存储模块（对标 Lab0511）。
使用 SQLite + CSV 存储训练过程和实验结果。
SQLite 无需安装额外服务，用法与 MySQL 相同，轻量且可移植。
"""

import os
import csv
import json
import sqlite3
import pandas as pd
from datetime import datetime


class ExperimentDataStore:
    """
    实验数据存储类。
    同时支持 SQLite 数据库和 CSV 文件两种存储方式。
    """

    def __init__(self, db_path='experiment_data.db', csv_dir='visualizations'):
        """
        Args:
            db_path: SQLite 数据库文件路径
            csv_dir: CSV 文件保存目录
        """
        self.db_path = db_path
        self.csv_dir = csv_dir
        os.makedirs(csv_dir, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """初始化 SQLite 数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 表1: 训练记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                epoch INTEGER,
                batch INTEGER,
                loss REAL,
                l1_loss REAL,
                learning_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 表2: 验证指标
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                epoch INTEGER,
                abs_rel REAL,
                sq_rel REAL,
                rmse REAL,
                rmse_log REAL,
                delta1 REAL,
                delta2 REAL,
                delta3 REAL,
                l1_loss REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 表3: 实验配置
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT UNIQUE,
                config_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 表4: 模型指标汇总
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                model_params INTEGER,
                best_l1_loss REAL,
                best_epoch INTEGER,
                r2_score REAL,
                mse REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        print(f"[data_storage] SQLite 数据库初始化完成: {self.db_path}")

    # ==================== 写入方法 ====================

    def save_training_record(self, exp_name, epoch, batch, loss, l1_loss, lr):
        """保存单条训练记录到 SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO training_records 
               (experiment_name, epoch, batch, loss, l1_loss, learning_rate)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (exp_name, epoch, batch, loss, l1_loss, lr)
        )
        conn.commit()
        conn.close()

    def save_validation_record(self, exp_name, epoch, metrics):
        """保存验证指标到 SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO validation_records
               (experiment_name, epoch, abs_rel, sq_rel, rmse, rmse_log,
                delta1, delta2, delta3, l1_loss)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (exp_name, epoch,
             metrics.get('abs_rel', 0), metrics.get('sq_rel', 0),
             metrics.get('rmse', 0), metrics.get('rmse_log', 0),
             metrics.get('delta1', 0), metrics.get('delta2', 0),
             metrics.get('delta3', 0), metrics.get('l1', 0))
        )
        conn.commit()
        conn.close()

    def save_config(self, exp_name, config_dict):
        """保存实验配置到 SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT OR IGNORE INTO experiment_config
               (experiment_name, config_json) VALUES (?, ?)''',
            (exp_name, json.dumps(config_dict, ensure_ascii=False))
        )
        conn.commit()
        conn.close()

    def save_model_summary(self, exp_name, model_params, best_l1, best_epoch,
                           r2_score=None, mse=None):
        """保存模型汇总信息到 SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO model_summary
               (experiment_name, model_params, best_l1_loss, best_epoch, r2_score, mse)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (exp_name, model_params, best_l1, best_epoch, r2_score, mse)
        )
        conn.commit()
        conn.close()

    # ==================== 导出到 CSV ====================

    def export_all_to_csv(self):
        """将所有表导出为 CSV 文件"""
        conn = sqlite3.connect(self.db_path)
        tables = ['training_records', 'validation_records',
                  'experiment_config', 'model_summary']

        for table in tables:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            if len(df) > 0:
                csv_path = os.path.join(self.csv_dir, f'{table}.csv')
                df.to_csv(csv_path, index=False)
                print(f"[data_storage] -> {csv_path} ({len(df)} 条记录)")

        conn.close()

    # ==================== 查询方法 ====================

    def query_training(self, exp_name=None, limit=100):
        """查询训练记录"""
        conn = sqlite3.connect(self.db_path)
        if exp_name:
            df = pd.read_sql(
                "SELECT * FROM training_records WHERE experiment_name=? ORDER BY id DESC LIMIT ?",
                conn, params=(exp_name, limit)
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM training_records ORDER BY id DESC LIMIT ?",
                conn, params=(limit,)
            )
        conn.close()
        return df

    def query_validation(self, exp_name=None):
        """查询验证记录"""
        conn = sqlite3.connect(self.db_path)
        if exp_name:
            df = pd.read_sql(
                "SELECT * FROM validation_records WHERE experiment_name=? ORDER BY epoch",
                conn, params=(exp_name,)
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM validation_records ORDER BY epoch",
                conn
            )
        conn.close()
        return df

    def close(self):
        """关闭数据库连接（SQLite 无需显式关闭，保留兼容性）"""
        pass


# ==================== 便利函数 ====================

def save_dataframe_csv(df, filename, save_dir='visualizations'):
    """
    将 pandas DataFrame 保存为 CSV 文件。
    
    Args:
        df: pandas DataFrame
        filename: 文件名（不含路径）
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"[data_storage] CSV 已保存: {path}")
    return path


def save_metrics_to_csv(metrics_dict, filename, save_dir='visualizations'):
    """
    将指标字典保存为 CSV 文件。
    
    Args:
        metrics_dict: 指标字典
        filename: 文件名
        save_dir: 保存目录
    """
    df = pd.DataFrame([metrics_dict])
    return save_dataframe_csv(df, filename, save_dir)


if __name__ == "__main__":
    print("Testing data_storage...")
    # 测试 SQLite 存储
    store = ExperimentDataStore(db_path='test_experiment.db')

    # 保存训练记录
    store.save_training_record('test_exp', 1, 0, 0.5, 0.3, 0.001)
    store.save_training_record('test_exp', 1, 1, 0.4, 0.25, 0.001)

    # 保存验证记录
    store.save_validation_record('test_exp', 1, {
        'abs_rel': 0.15, 'sq_rel': 0.5, 'rmse': 2.0,
        'rmse_log': 0.1, 'delta1': 0.8, 'delta2': 0.9,
        'delta3': 0.95, 'l1': 1.5
    })

    # 保存配置
    store.save_config('test_exp', {'epochs': 10, 'lr': 0.001})

    # 导出 CSV
    store.export_all_to_csv()

    # 查询
    df = store.query_training('test_exp')
    print(f"查询结果: {len(df)} 条记录")

    # 清理测试文件
    import os
    os.remove('test_experiment.db')
    print("Done!")