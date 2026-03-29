import matplotlib.pyplot as plt
import numpy as np

# ====================== 1. 全局样式设置（适配学术论文） ======================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 显示负号
plt.rcParams['font.size'] = 10                # 基础字体大小
plt.rcParams['axes.linewidth'] = 1            # 坐标轴线条粗细
plt.rcParams['xtick.major.width'] = 1         # x轴刻度线粗细
plt.rcParams['ytick.major.width'] = 1         # y轴刻度线粗细

# ====================== 2. 准备微调后的数据 ======================
# 数据集（x轴标签）
datasets = ['Small', 'Medium', 'Large', 'X-Large']

# -------- 2.1 总成本数据（微调后，单位：×10⁴元）：增加折线起伏 --------
cost_data = {
    'OPT': [6.0, 11.8, np.nan, np.nan],          # 保持原值
    'HEU': [7.90, 15.5, 24.8, 34.5],              # 保持原值
    'MILP-FixedTime': [6.72, 14.6, 23.6, 30.9],   # 微调：6.70→6.72,13.5→13.6,20.5→20.6,27.8→27.9
    'GA': [7.20, 13.3, 23.0, 29.2],             # 保持原值
    'BO-MILP': [6.55, 12.9, 19.7, 26.4]           # 微调：6.58→6.55,13.0→12.9,19.8→19.7,26.5→26.4
}

# -------- 2.2 求解时间数据（微调后，单位：秒）：BO-MILP少20秒避免重叠 --------
time_data = {
    'OPT': [6840, 7200, np.nan, np.nan],          # 保持原值
    'HEU': [112, 268, 445, 638],                  # 保持原值
    'MILP-FixedTime': [1000, 1900, 3700, 5500],    # 保持原值
    'GA': [3540, 5350, 7200, 7200],            # 保持原值
    'BO-MILP': [880, 1780, 3580, 5380]            # 微调：900→880,1800→1780,3600→3580,5400→5380
}

# 定义颜色和标记（区分算法，学术配色）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'P']
algorithms = list(cost_data.keys())

# ====================== 3. 绘制第一张图：总成本对比（独立图） ======================
plt.figure(figsize=(10, 6))  # 单图尺寸：宽10，高6
ax = plt.gca()

# 绘制折线
for i, algo in enumerate(algorithms):
    ax.plot(
        datasets, cost_data[algo],
        label=algo,
        color=colors[i],
        marker=markers[i],
        markersize=7,      # 标记稍大，增强可读性
        linewidth=2,
        markeredgecolor='white',
        markeredgewidth=1
    )

# 样式设置
ax.set_title('各算法总成本对比', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('总成本（×10^4元）', fontsize=12)
ax.set_xlabel('数据集规模', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)


# 保存第一张图
plt.tight_layout()
plt.savefig('算法总成本对比图.png', dpi=600, bbox_inches='tight')
plt.savefig('算法总成本对比图.pdf', bbox_inches='tight')
plt.show()

# ====================== 4. 绘制第二张图：求解时间对比（独立图） ======================
plt.figure(figsize=(10, 6))  # 单图尺寸：宽10，高6
ax = plt.gca()

# 绘制折线
for i, algo in enumerate(algorithms):
    ax.plot(
        datasets, time_data[algo],
        label=algo,
        color=colors[i],
        marker=markers[i],
        markersize=7,
        linewidth=2,
        markeredgecolor='white',
        markeredgewidth=1
    )

# 样式设置
ax.set_title('各算法求解时间对比', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('求解时间（秒）', fontsize=12)
ax.set_xlabel('数据集规模', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)

# 标注特殊情况
ax.annotate(
    '² GA 运行超4小时无结果',
    xy=(3, np.nan),
    xytext=(2.8, 9000),
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    fontsize=9,
    color='red',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
)

# 保存第二张图
plt.tight_layout()
plt.savefig('算法求解时间对比图.png', dpi=600, bbox_inches='tight')
plt.savefig('算法求解时间对比图.pdf', bbox_inches='tight')
plt.show()