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


class FogComputingVisualizer:
    def __init__(self, root_dir=None):
        """初始化可视化工具"""
        # 如果未指定根目录，默认使用上级目录中的output_lxl
        if root_dir is None:
            # 获取当前脚本所在目录（code_lxl）
            current_dir = Path(__file__).parent
            # 上级目录中寻找output_lxl
            self.root_dir = current_dir.parent / "output_lxl"
        else:
            self.root_dir = Path(root_dir)

        # 检查目录是否存在
        if not self.root_dir.exists():
            raise FileNotFoundError(f"找不到数据目录: {self.root_dir}")

        self.algorithms = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        self.data = {}  # 存储所有加载的数据

    def load_data(self):
        """加载所有算法的训练和评估数据"""
        for algo in self.algorithms:
            algo_dir = self.root_dir / algo
            self.data[algo] = {}

            # 加载训练数据
            train_dir = algo_dir / "train"
            if train_dir.exists():
                train_metrics_path = train_dir / "training_metrics.npy"
                if train_metrics_path.exists():
                    train_data = np.load(train_metrics_path, allow_pickle=True).item()
                    self.data[algo]["train"] = {
                        "rewards": train_data.get("episode_rewards", []),
                        "latencies": train_data.get("episode_latencies", []),
                        "energies": train_data.get("episode_energies", []),
                        "successful_tasks": train_data.get("episode_successful_Tasks", [])
                    }

            # 加载评估数据
            eval_dir = algo_dir / "evaluate"
            if eval_dir.exists():
                self.data[algo]["eval"] = {}

                reward_path = eval_dir / "evaluation_rewards.npy"
                if reward_path.exists():
                    self.data[algo]["eval"]["rewards"] = np.load(reward_path)

                latency_path = eval_dir / "evaluation_latencies.npy"
                if latency_path.exists():
                    self.data[algo]["eval"]["latencies"] = np.load(latency_path)

                energy_path = eval_dir / "evaluation_energy.npy"
                if energy_path.exists():
                    self.data[algo]["eval"]["energies"] = np.load(energy_path)

        print(f"已加载 {len(self.algorithms)} 个算法的数据")
        return self

    def plot_training_curve(self, algo, metric="rewards", smooth_window=10, save_path=None):
        """绘制单个算法的训练曲线"""
        if algo not in self.data or "train" not in self.data[algo] or metric not in self.data[algo]["train"]:
            print(f"警告：没有找到 {algo} 的训练{metric}数据")
            return

        data = self.data[algo]["train"][metric]
        episodes = range(1, len(data) + 1)

        # 数据平滑处理
        if smooth_window > 1 and len(data) >= smooth_window:
            smoothed_data = np.convolve(data, np.ones(smooth_window) / smooth_window, mode='valid')
            smoothed_episodes = range(smooth_window, len(data) + 1)
        else:
            smoothed_data = data
            smoothed_episodes = episodes

        # 设置图表
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, data, alpha=0.3, label="原始数据")
        plt.plot(smoothed_episodes, smoothed_data, label=f"平滑数据 (窗口={smooth_window})")

        metric_names = {
            "rewards": "奖励",
            "latencies": "延迟",
            "energies": "能耗",
            "successful_tasks": "成功任务数"
        }

        # 明确指定字体
        font_props = FontProperties(family=available_fonts[0]) if available_fonts else FontProperties()

        plt.title(f"{algo} 训练过程{metric_names[metric]}曲线", fontproperties=font_props, fontsize=14)
        plt.xlabel("训练轮次 (Episode)", fontproperties=font_props, fontsize=12)
        plt.ylabel(metric_names[metric], fontproperties=font_props, fontsize=12)
        plt.legend(prop=font_props)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至 {save_path}")

        plt.show()

    def plot_evaluation_metrics(self, algo, save_path=None):
        """绘制单个算法的评估指标"""
        if algo not in self.data or "eval" not in self.data[algo]:
            print(f"警告：没有找到 {algo} 的评估数据")
            return

        eval_data = self.data[algo]["eval"]
        metrics = [m for m in ["rewards", "latencies", "energies"] if m in eval_data]

        if not metrics:
            print(f"警告：{algo} 没有可用的评估指标数据")
            return

        # 设置图表
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        metric_names = {
            "rewards": "奖励",
            "latencies": "延迟",
            "energies": "能耗"
        }

        # 明确指定字体
        font_props = FontProperties(family=available_fonts[0]) if available_fonts else FontProperties()

        for i, metric in enumerate(metrics):
            data = eval_data[metric]
            episodes = range(1, len(data) + 1)

            axes[i].plot(episodes, data, marker='o', linestyle='-', markersize=4)
            axes[i].set_title(f"{metric_names[metric]}", fontproperties=font_props, fontsize=12)
            axes[i].set_xlabel("评估轮次", fontproperties=font_props, fontsize=10)
            axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴为整数

            # 添加平均值参考线
            mean_val = np.mean(data)
            axes[i].axhline(mean_val, color='r', linestyle='--', alpha=0.7,
                            label=f"平均值: {mean_val:.2f}")
            axes[i].legend(prop=font_props)

        fig.suptitle(f"{algo} 评估指标", fontproperties=font_props, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至 {save_path}")

        plt.show()

    def plot_training_comparison(self, metric="rewards", smooth_window=10, save_path=None):
        """对比所有算法的训练指标"""
        plt.figure(figsize=(12, 7))

        metric_names = {
            "rewards": "奖励",
            "latencies": "延迟",
            "energies": "能耗",
            "successful_tasks": "成功任务数"
        }

        # 明确指定字体
        font_props = FontProperties(family=available_fonts[0]) if available_fonts else FontProperties()

        for algo in self.algorithms:
            if "train" not in self.data[algo] or metric not in self.data[algo]["train"]:
                print(f"警告：跳过 {algo}，没有找到训练{metric_names[metric]}数据")
                continue

            data = self.data[algo]["train"][metric]
            if len(data) == 0:
                continue

            # 数据平滑
            if smooth_window > 1 and len(data) >= smooth_window:
                smoothed_data = np.convolve(data, np.ones(smooth_window) / smooth_window, mode='valid')
                episodes = range(smooth_window, len(data) + 1)
            else:
                smoothed_data = data
                episodes = range(1, len(data) + 1)

            plt.plot(episodes, smoothed_data, label=algo)

        plt.title(f"不同算法训练过程{metric_names[metric]}对比", fontproperties=font_props, fontsize=14)
        plt.xlabel("训练轮次 (Episode)", fontproperties=font_props, fontsize=12)
        plt.ylabel(metric_names[metric], fontproperties=font_props, fontsize=12)
        plt.legend(prop=font_props)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图表已保存至 {save_path}")

        plt.show()

    def plot_evaluation_comparison(self, metric="rewards", save_path=None):
        """对比所有算法的评估指标"""
        # 收集所有算法的评估数据
        eval_data = {}
        for algo in self.algorithms:
            if "eval" in self.data[algo] and metric in self.data[algo]["eval"]:
                data = self.data[algo]["eval"][metric]
                if len(data) > 0:
                    eval_data[algo] = data

        if not eval_data:
            print(f"警告：没有找到任何算法的评估{metric}数据")
            return

        metric_names = {
            "rewards": "奖励",
            "latencies": "延迟",
            "energies": "能耗"
        }

        # 明确指定字体
        font_props = FontProperties(family=available_fonts[0]) if available_fonts else FontProperties()

        # 绘制箱线图比较分布
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[eval_data[algo] for algo in eval_data.keys()])
        plt.xticks(range(len(eval_data)), eval_data.keys(), fontproperties=font_props)
        plt.title(f"不同算法评估{metric_names[metric]}分布对比", fontproperties=font_props, fontsize=14)
        plt.ylabel(metric_names[metric], fontproperties=font_props, fontsize=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图表已保存至 {save_path}")

        plt.show()

        # 绘制平均值对比图
        plt.figure(figsize=(10, 6))
        averages = [np.mean(eval_data[algo]) for algo in eval_data.keys()]
        sns.barplot(x=list(eval_data.keys()), y=averages)

        # 在柱状图上标注平均值
        for i, avg in enumerate(averages):
            plt.text(i, avg + np.max(averages) * 0.01, f"{avg:.2f}",
                     ha='center', fontsize=10, fontproperties=font_props)

        plt.title(f"不同算法评估{metric_names[metric]}平均值对比", fontproperties=font_props, fontsize=14)
        plt.ylabel(f"平均{metric_names[metric]}", fontproperties=font_props, fontsize=12)
        plt.xticks(range(len(eval_data)), eval_data.keys(), fontproperties=font_props)
        plt.tight_layout()

        if save_path:
            base, ext = os.path.splitext(save_path)
            avg_save_path = f"{base}_average{ext}"
            plt.savefig(avg_save_path, dpi=300, bbox_inches='tight')
            print(f"平均值对比图表已保存至 {avg_save_path}")

        plt.show()

    def generate_all_visualizations(self, output_dir="visualizations"):
        """生成所有可能的可视化图表并保存"""
        # 创建保存目录（在上级目录中，与output_lxl同级）
        output_dir = Path(__file__).parent.parent / output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 为每个算法生成训练和评估可视化
        for algo in self.algorithms:
            algo_dir = os.path.join(output_dir, algo)
            os.makedirs(algo_dir, exist_ok=True)

            # 训练指标
            for metric in ["rewards", "latencies", "energies", "successful_tasks"]:
                self.plot_training_curve(
                    algo,
                    metric=metric,
                    save_path=os.path.join(algo_dir, f"training_{metric}.png")
                )

            # 评估指标
            self.plot_evaluation_metrics(
                algo,
                save_path=os.path.join(algo_dir, "evaluation_metrics.png")
            )

        # 生成算法对比可视化
        comparison_dir = os.path.join(output_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)

        # 训练对比
        for metric in ["rewards", "latencies", "energies", "successful_tasks"]:
            self.plot_training_comparison(
                metric=metric,
                save_path=os.path.join(comparison_dir, f"training_{metric}_comparison.png")
            )

        # 评估对比
        for metric in ["rewards", "latencies", "energies"]:
            self.plot_evaluation_comparison(
                metric=metric,
                save_path=os.path.join(comparison_dir, f"evaluation_{metric}_comparison.png")
            )

        print(f"所有可视化图表已保存至 {output_dir} 目录")


if __name__ == "__main__":
    try:
        # 创建可视化工具实例，自动寻找上级目录中的output_lxl
        visualizer = FogComputingVisualizer()

        # 加载数据
        visualizer.load_data()

        # 生成所有可视化图表
        visualizer.generate_all_visualizations()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确认output_lxl文件夹与code_lxl文件夹在同一级目录下")
    except Exception as e:
        print(f"发生错误: {e}")
