import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import findfont, FontProperties


# ------------------------------
# 解决中文显示问题的字体设置
# ------------------------------
def check_chinese_fonts():
    """检查系统中可用的中文字体"""
    # 常见中文字体列表
    chinese_fonts = [
        "SimHei", "WenQuanYi Micro Hei", "Heiti TC",
        "Microsoft YaHei", "SimSun", "FangSong", "KaiTi"
    ]

    available_fonts = []
    for font in chinese_fonts:
        try:
            # 尝试查找字体
            font_path = findfont(FontProperties(family=font))
            available_fonts.append((font, font_path))
        except:
            continue

    if not available_fonts:
        print("警告：未检测到任何可用的中文字体！")
        print("请安装以下任意以下任意一种中文字体：")
        print(", ".join(chinese_fonts))
    else:
        print("检测到可用的中文字体：")
        for i, (font, path) in enumerate(available_fonts, 1):
            print(f"{i}. {font} - {path}")

    return [font for font, _ in available_fonts]


# 检查并设置可用的中文字体
available_fonts = check_chinese_fonts()
if available_fonts:
    # 设置字体，优先使用第一个可用字体
    plt.rcParams["font.family"] = available_fonts
else:
    # 如果没有中文字体，使用默认字体并提示
    print("将使用默认字体，可能无法正确显示中文")

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
sns.set_style("whitegrid")