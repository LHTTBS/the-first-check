import matplotlib.pyplot as plt
import numpy as np
import re  # 用于解析TXT日志中的数值

# -------------------------- 1. TXT日志解析函数（核心新增）--------------------------
def parse_training_log(txt_path):
    """
    从DMINT模型训练日志TXT文件中提取核心指标
    :param txt_path: TXT日志文件路径（如'logs-34.txt'）
    :return: 字典格式的所有训练指标（epoch、train_loss、val_loss、各任务指标等）
    """
    # 初始化存储列表
    epochs = []
    train_loss = []
    val_loss = []
    learning_rate = []
    stance_acc = []
    stance_f1 = []
    harm_acc = []
    harm_f1 = []
    fair_acc = []
    fair_f1 = []
    intent_macro_f1 = []
    intent_political_f1 = []
    intent_public_f1 = []

    # 读取TXT日志
    with open(txt_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # 正则表达式匹配每个Epoch的完成数据（关键：匹配"Epoch X 完成:"后的所有指标）
    epoch_pattern = r"Epoch (\d+)/\d+.*?Epoch \1 完成:.*?训练损失: ([\d.]+).*?验证损失: ([\d.]+).*?当前学习率: ([\d.e+-]+).*?stance_accuracy: ([\d.]+).*?stance_f1: ([\d.]+).*?harmfulness_accuracy: ([\d.]+).*?harmfulness_f1: ([\d.]+).*?fairness_accuracy: ([\d.]+).*?fairness_f1: ([\d.]+).*?intent_macro_f1: ([\d.]+).*?intent_Political_f1: ([\d.]+).*?intent_Public_f1: ([\d.]+)"
    matches = re.findall(epoch_pattern, log_content, re.DOTALL)  # re.DOTALL让.匹配换行符

    # 提取匹配到的数值并转换类型
    for match in matches:
        epochs.append(int(match[0]))  # Epoch序号
        train_loss.append(float(match[1]))  # 训练损失
        val_loss.append(float(match[2]))  # 验证损失
        learning_rate.append(float(match[3]))  # 学习率（支持科学计数法）
        stance_acc.append(float(match[4]))  # Stance准确率
        stance_f1.append(float(match[5]))  # Stance F1
        harm_acc.append(float(match[6]))  # Harmfulness准确率
        harm_f1.append(float(match[7]))  # Harmfulness F1
        fair_acc.append(float(match[8]))  # Fairness准确率
        fair_f1.append(float(match[9]))  # Fairness F1
        intent_macro_f1.append(float(match[10]))  # Intent Macro F1
        intent_political_f1.append(float(match[11]))  # Intent Political F1
        intent_public_f1.append(float(match[12]))  # Intent Public F1

    # 返回解析后的字典
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate,
        'stance_acc': stance_acc,
        'stance_f1': stance_f1,
        'harm_acc': harm_acc,
        'harm_f1': harm_f1,
        'fair_acc': fair_acc,
        'fair_f1': fair_f1,
        'intent_macro_f1': intent_macro_f1,
        'intent_political_f1': intent_political_f1,
        'intent_public_f1': intent_public_f1
    }

# -------------------------- 2. 解析你的TXT日志（只需修改这里的路径）--------------------------
# 替换为你的logs-34.txt实际路径（如放在当前文件夹，直接写文件名；否则写完整路径如'D:/data/logs-34.txt'）
log_data = parse_training_log(txt_path='outputs\logs-34.txt')

# 从解析结果中提取数据（后续可视化直接用）
epochs = log_data['epochs']
train_loss = log_data['train_loss']
val_loss = log_data['val_loss']
learning_rate = log_data['learning_rate']
stance_acc = log_data['stance_acc']
stance_f1 = log_data['stance_f1']
harm_acc = log_data['harm_acc']
harm_f1 = log_data['harm_f1']
fair_acc = log_data['fair_acc']
fair_f1 = log_data['fair_f1']
intent_macro_f1 = log_data['intent_macro_f1']
intent_political_f1 = log_data['intent_political_f1']
intent_public_f1 = log_data['intent_public_f1']

# -------------------------- 3. 可视化绘图（与之前一致，数据来源改为TXT解析结果）--------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
fig, axes = plt.subplots(3, 2, figsize=(16, 12))  # 3行2列子图，适配多指标展示
fig.suptitle('优化版DMINT模型训练过程可视化（来自logs-34.txt）', fontsize=16, fontweight='bold')

# 子图1：训练损失 vs 验证损失（核心过拟合判断依据）
axes[0, 0].plot(epochs, train_loss, label='训练损失', color='#1f77b4', linewidth=2, marker='o', markersize=3)
axes[0, 0].plot(epochs, val_loss, label='验证损失', color='#ff7f0e', linewidth=2, marker='s', markersize=3)
axes[0, 0].set_title('训练损失与验证损失趋势', fontweight='bold')
axes[0, 0].set_xlabel('Epoch（训练轮次）')
axes[0, 0].set_ylabel('损失值')
axes[0, 0].legend(loc='upper right')
axes[0, 0].grid(alpha=0.3)  # 网格线（提高可读性）

# 子图2：学习率调度变化（查看学习率是否合理下降）
axes[0, 1].plot(epochs, learning_rate, color='#2ca02c', linewidth=2, marker='^', markersize=3)
axes[0, 1].set_title('学习率变化曲线', fontweight='bold')
axes[0, 1].set_xlabel('Epoch（训练轮次）')
axes[0, 1].set_ylabel('学习率')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_yscale('log')  # 对数尺度（科学计数法学习率更易观察）

# 子图3：Stance任务指标（立场分类任务）
axes[1, 0].plot(epochs, stance_acc, label='Stance准确率', color='#d62728', linewidth=2, marker='o', markersize=3)
axes[1, 0].plot(epochs, stance_f1, label='Stance F1分数', color='#9467bd', linewidth=2, marker='s', markersize=3)
axes[1, 0].set_title('Stance任务性能指标', fontweight='bold')
axes[1, 0].set_xlabel('Epoch（训练轮次）')
axes[1, 0].set_ylabel('分数（0-1）')
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_ylim(0.6, 0.9)  # 固定y轴范围（突出变化细节）

# 子图4：Harmfulness任务指标（危害性判断任务）
axes[1, 1].plot(epochs, harm_acc, label='Harmfulness准确率', color='#8c564b', linewidth=2, marker='o', markersize=3)
axes[1, 1].plot(epochs, harm_f1, label='Harmfulness F1分数', color='#e377c2', linewidth=2, marker='s', markersize=3)
axes[1, 1].set_title('Harmfulness任务性能指标', fontweight='bold')
axes[1, 1].set_xlabel('Epoch（训练轮次）')
axes[1, 1].set_ylabel('分数（0-1）')
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim(0.6, 0.8)  # 适配该任务的分数范围

# 子图5：Fairness任务指标（公平性判断任务）
axes[2, 0].plot(epochs, fair_acc, label='Fairness准确率', color='#7f7f7f', linewidth=2, marker='o', markersize=3)
axes[2, 0].plot(epochs, fair_f1, label='Fairness F1分数', color='#bcbd22', linewidth=2, marker='s', markersize=3)
axes[2, 0].set_title('Fairness任务性能指标', fontweight='bold')
axes[2, 0].set_xlabel('Epoch（训练轮次）')
axes[2, 0].set_ylabel('分数（0-1）')
axes[2, 0].legend(loc='lower right')
axes[2, 0].grid(alpha=0.3)
axes[2, 0].set_ylim(0.6, 0.9)  # 固定y轴范围

# 子图6：Intent任务核心F1（意图分类任务，重点关注Political/Public）
axes[2, 1].plot(epochs, intent_macro_f1, label='Intent Macro F1', color='#17becf', linewidth=2, marker='o', markersize=3)
axes[2, 1].plot(epochs, intent_political_f1, label='Intent-Political F1', color='#ffbb78', linewidth=2, marker='s', markersize=3)
axes[2, 1].plot(epochs, intent_public_f1, label='Intent-Public F1', color='#98df8a', linewidth=2, marker='^', markersize=3)
axes[2, 1].set_title('Intent任务核心F1分数', fontweight='bold')
axes[2, 1].set_xlabel('Epoch（训练轮次）')
axes[2, 1].set_ylabel('F1分数（0-1）')
axes[2, 1].legend(loc='lower right')
axes[2, 1].grid(alpha=0.3)
axes[2, 1].set_ylim(0.3, 0.9)  # 适配Macro F1的低分值范围

# 调整子图间距，避免标签重叠
plt.tight_layout()
# 保存图片（高清300dpi，支持缩放）
plt.savefig('DMINT_training_analysis_from_TXT.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印解析结果摘要（验证数据是否正确提取）
print(f"✅ 成功从TXT日志中提取 {len(epochs)} 个Epoch的训练数据")
print(f"📊 训练损失变化：{train_loss[0]:.4f} → {train_loss[-1]:.4f}（下降{((train_loss[0]-train_loss[-1])/train_loss[0]*100):.1f}%）")
print(f"📊 验证损失变化：{val_loss[0]:.4f} → {val_loss[-1]:.4f}（上升{((val_loss[-1]-val_loss[0])/val_loss[0]*100):.1f}%）")
print(f"🏆 最佳Stance F1：{max(stance_f1):.4f}（Epoch {epochs[stance_f1.index(max(stance_f1))]}）")
print(f"🏆 最佳Intent Macro F1：{max(intent_macro_f1):.4f}（Epoch {epochs[intent_macro_f1.index(max(intent_macro_f1))]}）")
print(f"🖼️  可视化图表已保存为：DMINT_training_analysis_from_TXT.png")