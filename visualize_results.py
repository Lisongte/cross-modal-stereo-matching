"""
可视化入口脚本 - 直接调用 data_analysis/visualize_results.py 的 main 函数。

用法:
    python visualize_results.py
"""

import os
import sys

# 确保能导入 data_analysis 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # 调用 data_analysis 中的可视化模块
    from data_analysis.visualize_results import main
    main()