#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务模型训练日志解析与可视化工具
直接从txt日志提取数据并生成完整分析报告
"""

import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# ==================== 中文设置 ====================
import matplotlib
font_candidates = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'STHeiti', 'Arial Unicode MS']
available_font = None
for font in font_candidates:
    try:
        matplotlib.font_manager.findfont(font)
        available_font = font
        break
    except:
        continue

if available_font:
    plt.rcParams['font.sans-serif'] = [available_font]
    print(f"✅ 已设置中文字体: {available_font}")
else:
    print("⚠️  未找到合适的中文字体，可能无法显示中文")
    
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
sns.set_style("whitegrid", {'axes.grid': True, 'grid.alpha': 0.3})

# ==================== 配置区域 ====================
LOG_FILE = "outputs\logs-34.txt"  # 你的txt文件名，可修改
OUTPUT_DIR = "outputs"  # 输出目录

# ==================== 日志解析模块 ====================
def parse_training_log(txt_path):
    """
    解析训练日志txt文件，提取结构化数据
    支持科学计数法的学习率格式
    """
    print(f"📂 正在解析日志文件: {txt_path}")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按epoch分割
    epoch_blocks = re.split(r'=+', content)
    history = {
        "epochs": [], "train_loss": [], "val_loss": [], "learning_rate": [],
        "stance_acc": [], "stance_f1": [], "harmfulness_acc": [], "harmfulness_f1": [],
        "fairness_acc": [], "fairness_f1": [], "intent_em": [], "intent_macro_f1": [],
        "intent_political_f1": [], "intent_economic_f1": [],
        "intent_psychological_f1": [], "intent_public_f1": []
    }
    
    epoch_count = 0
    
    for block in epoch_blocks:
        if "Epoch" not in block or "完成:" not in block:
            continue
            
        # 提取epoch编号
        epoch_match = re.search(r'Epoch (\d+)', block)
        if not epoch_match:
            continue
        
        epoch = int(epoch_match.group(1))
        
        # 提取指标（支持科学计数法）
        patterns = {
            "train_loss": r"训练损失: ([\d.]+)",
            "val_loss": r"验证损失: ([\d.]+)",
            "learning_rate": r"当前学习率: ([\de\-\.]+)",
            "stance_acc": r"stance_accuracy: ([\d.]+)",
            "stance_f1": r"stance_f1: ([\d.]+)",
            "harmfulness_acc": r"harmfulness_accuracy: ([\d.]+)",
            "harmfulness_f1": r"harmfulness_f1: ([\d.]+)",
            "fairness_acc": r"fairness_accuracy: ([\d.]+)",
            "fairness_f1": r"fairness_f1: ([\d.]+)",
            "intent_em": r"intent_exact_match: ([\d.]+)",
            "intent_macro_f1": r"intent_macro_f1: ([\d.]+)",
            "intent_political_f1": r"intent_Political_f1: ([\d.]+)",
            "intent_economic_f1": r"intent_Economic_f1: ([\d.]+)",
            "intent_psychological_f1": r"intent_Psychological_f1: ([\d.]+)",
            "intent_public_f1": r"intent_Public_f1: ([\d.]+)"
        }
        
        epoch_data = {"epoch": epoch}
        for key, pattern in patterns.items():
            match = re.search(pattern, block)
            if match:
                epoch_data[key] = float(match.group(1))
            else:
                epoch_data[key] = 0.0
        
        # 添加到history
        for key in history.keys():
            if key != "epochs":
                history[key].append(epoch_data[key])
        history["epochs"].append(epoch)
        epoch_count += 1
    
    if epoch_count == 0:
        print("❌ 未能提取到有效数据，请检查日志格式")
        return None
    
    print(f"✅ 成功解析 {epoch_count} 个epoch")
    return history

# ==================== 绘图函数 ====================
def plot_training_analysis(history):
    """生成2x4布局的分析图表，严格复现原始图表"""
    epochs = history['epochs']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('多任务模型训练过程综合分析', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. 训练与验证损失（左上图）
    axes[0,0].plot(epochs, history['train_loss'], 'b-', linewidth=2.5, label='训练损失', marker='o', markersize=3)
    axes[0,0].plot(epochs, history['val_loss'], 'r-', linewidth=2.5, label='验证损失', marker='s', markersize=3)
    axes[0,0].set_title('训练与验证损失', fontweight='bold', fontsize=12)
    axes[0,0].set_xlabel('Epochs', fontsize=10)
    axes[0,0].set_ylabel('Loss', fontsize=10)
    axes[0,0].legend(loc='upper right', fontsize=9)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 学习率调度（右上图）
    axes[0,1].plot(epochs, history['learning_rate'], 'g-', linewidth=2.5, marker='^', markersize=3)
    axes[0,1].set_title('学习率调度', fontweight='bold', fontsize=12)
    axes[0,1].set_xlabel('Epochs', fontsize=10)
    axes[0,1].set_ylabel('Learning Rate', fontsize=10)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_yscale('log')
    
    # 3. 立场任务（中左）
    axes[0,2].plot(epochs, history['stance_acc'], 'c-', linewidth=2.5, label='准确率', marker='o', markersize=3)
    axes[0,2].plot(epochs, history['stance_f1'], 'm-', linewidth=2.5, label='F1-Score', marker='s', markersize=3)
    axes[0,2].set_title('立场检测任务', fontweight='bold', fontsize=12)
    axes[0,2].set_xlabel('Epochs', fontsize=10)
    axes[0,2].set_ylabel('Score', fontsize=10)
    axes[0,2].legend(loc='lower right', fontsize=9)
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_ylim(0.6, 0.85)
    
    # 4. 有害性任务（中右）
    axes[0,3].plot(epochs, history['harmfulness_acc'], 'c-', linewidth=2.5, label='准确率', marker='o', markersize=3)
    axes[0,3].plot(epochs, history['harmfulness_f1'], 'm-', linewidth=2.5, label='F1-Score', marker='s', markersize=3)
    axes[0,3].set_title('有害性检测任务', fontweight='bold', fontsize=12)
    axes[0,3].set_xlabel('Epochs', fontsize=10)
    axes[0,3].set_ylabel('Score', fontsize=10)
    axes[0,3].legend(loc='lower right', fontsize=9)
    axes[0,3].grid(True, alpha=0.3)
    axes[0,3].set_ylim(0.6, 0.85)
    
    # 5. 公平性任务（左下）
    axes[1,0].plot(epochs, history['fairness_f1'], color='#ff7f0e', linewidth=2.5, marker='D', markersize=3)
    axes[1,0].set_title('公平性任务 F1-Score', fontweight='bold', fontsize=12)
    axes[1,0].set_xlabel('Epochs', fontsize=10)
    axes[1,0].set_ylabel('F1 Score', fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(0.6, 0.85)
    
    # 6. 意图识别总体（下中左）
    axes[1,1].plot(epochs, history['intent_em'], 'purple', linewidth=2.5, label='Exact Match', marker='o', markersize=3)
    axes[1,1].plot(epochs, history['intent_macro_f1'], 'brown', linewidth=2.5, label='Macro F1', marker='s', markersize=3)
    axes[1,1].set_title('意图识别任务（总体）', fontweight='bold', fontsize=12)
    axes[1,1].set_xlabel('Epochs', fontsize=10)
    axes[1,1].set_ylabel('Score', fontsize=10)
    axes[1,1].legend(loc='lower right', fontsize=9)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(0.15, 0.45)
    
    # 7. 意图识别各类别（下中右）
    axes[1,2].plot(epochs, history['intent_political_f1'], 'r-', linewidth=2, label='政治', marker='o', markersize=2)
    axes[1,2].plot(epochs, history['intent_economic_f1'], 'g-', linewidth=2, label='经济', marker='s', markersize=2)
    axes[1,2].plot(epochs, history['intent_psychological_f1'], 'b-', linewidth=2, label='心理', marker='^', markersize=2)
    axes[1,2].plot(epochs, history['intent_public_f1'], 'orange', linewidth=2, label='公众', marker='d', markersize=2)
    axes[1,2].set_title('意图识别 - 各类别 F1', fontweight='bold', fontsize=12)
    axes[1,2].set_xlabel('Epochs', fontsize=10)
    axes[1,2].set_ylabel('F1 Score', fontsize=10)
    axes[1,2].legend(loc='lower right', fontsize=8, ncol=2)
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim(0.15, 0.7)
    
    # 8. 性能总结表格（右下）
    axes[1,3].axis('off')
    
    summary_data = [
        ['任务', '最终得分', '提升幅度'],
        ['立场检测', f"{history['stance_f1'][-1]:.3f}", f"+{history['stance_f1'][-1] - history['stance_f1'][0]:.3f}"],
        ['有害性检测', f"{history['harmfulness_f1'][-1]:.3f}", f"+{history['harmfulness_f1'][-1] - history['harmfulness_f1'][0]:.3f}"],
        ['公平性', f"{history['fairness_f1'][-1]:.3f}", f"+{history['fairness_f1'][-1] - history['fairness_f1'][0]:.3f}"],
        ['意图识别(EM)', f"{history['intent_em'][-1]:.3f}", f"+{history['intent_em'][-1] - history['intent_em'][0]:.3f}"],
        ['意图识别(F1)', f"{history['intent_macro_f1'][-1]:.3f}", f"+{history['intent_macro_f1'][-1] - history['intent_macro_f1'][0]:.3f}"]
    ]
    
    table = axes[1,3].table(cellText=summary_data, cellLoc='center', loc='center',
                           colWidths=[0.35, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # 设置表头样式
    for i in range(len(summary_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white')
    
    axes[1,3].set_title('任务性能总结', fontweight='bold', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("📊 图表已保存: training_analysis.png")
    plt.close()

# ==================== 数据保存 ====================
def save_all_data(history):
    """保存解析后的所有数据"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # CSV格式
    df = pd.DataFrame(history)
    df.to_csv(f"{OUTPUT_DIR}/training_history.csv", index=False, encoding='utf-8')
    
    # JSON格式
    with open(f"{OUTPUT_DIR}/training_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    # 最佳模型信息
    best_epoch = history['val_loss'].index(min(history['val_loss']))
    best_info = {
        'best_epoch': best_epoch + 1,
        'val_loss': history['val_loss'][best_epoch],
        'train_loss': history['train_loss'][best_epoch],
        'stance_f1': history['stance_f1'][best_epoch],
        'harmfulness_f1': history['harmfulness_f1'][best_epoch],
        'fairness_f1': history['fairness_f1'][best_epoch],
        'intent_em': history['intent_em'][best_epoch],
        'intent_macro_f1': history['intent_macro_f1'][best_epoch],
        'intent_political_f1': history['intent_political_f1'][best_epoch],
        'intent_economic_f1': history['intent_economic_f1'][best_epoch],
        'intent_psychological_f1': history['intent_psychological_f1'][best_epoch],
        'intent_public_f1': history['intent_public_f1'][best_epoch]
    }
    
    with open(f"{OUTPUT_DIR}/best_model.json", 'w') as f:
        json.dump(best_info, f, indent=2)
    
    print(f"💾 数据已保存至 {OUTPUT_DIR}/ 目录")
    return best_info

# ==================== 分析报告 ====================
def generate_comprehensive_report(history, best_info):
    """生成综合文字报告"""
    print("\n" + "="*60)
    print("📊 训练分析报告")
    print("="*60)
    
    print(f"\n🎯 训练概况:")
    print(f"   总轮数: {len(history['epochs'])}")
    print(f"   最佳轮次: 第 {best_info['best_epoch']} 轮")
    print(f"   最佳验证损失: {best_info['val_loss']:.4f}")
    
    print(f"\n📈 最终性能:")
    for task in ['stance_f1', 'harmfulness_f1', 'fairness_f1', 'intent_macro_f1']:
        print(f"   {task}: {history[task][-1]:.4f}")
    
    print(f"\n🚨 严重问题检测:")
    zero_count = sum(1 for f1 in history['intent_economic_f1'] if f1 < 0.01)
    if zero_count > 5:
        print(f"   🔴 Economic意图F1为0的轮数: {zero_count}/{len(history['epochs'])}")
        print("   → 可能原因: 验证集无正样本/标签编码错误/损失权重异常")
    
    zero_count = sum(1 for f1 in history['intent_psychological_f1'] if f1 < 0.01)
    if zero_count > 5:
        print(f"   🔴 Psychological意图F1为0的轮数: {zero_count}/{len(history['epochs'])}")
        print("   → 可能原因: 同上")
    
    gap = history['val_loss'][-1] - history['train_loss'][-1]
    if gap > 0.3:
        print(f"   🟡 过拟合风险: 验证损失比训练损失高 {gap:.3f}")
    
    print(f"\n💡 建议:")
    if zero_count > 5:
        print("   1. 立即检查验证集标签分布")
        print("   2. 检查intent子任务的标签编码逻辑")
        print("   3. 检查加权损失函数的权重计算")
    else:
        print("   模型训练正常，可以在测试集上评估性能")

# ==================== 主入口 ====================
def main():
    """主流程"""
    print("="*60)
    print("🚀 训练日志分析工具")
    print("="*60)
    
    # 检查文件
    if not Path(LOG_FILE).exists():
        print(f"❌ 文件不存在: {LOG_FILE}")
        print("请将日志文件放在当前目录，或修改LOG_FILE变量")
        sys.exit(1)
    
    # 解析日志
    history = parse_training_log(LOG_FILE)
    if history is None:
        sys.exit(1)
    
    # 保存数据
    best_info = save_all_data(history)
    
    # 生成图表
    plot_training_analysis(history)
    
    # 生成报告
    generate_comprehensive_report(history, best_info)
    
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60)
    print("📄 输出文件:")
    print("   - training_analysis.png (图表)")
    print(f"   - {OUTPUT_DIR}/training_history.csv (数据)")
    print(f"   - {OUTPUT_DIR}/training_history.json (数据)")
    print(f"   - {OUTPUT_DIR}/best_model.json (最佳模型信息)")

if __name__ == "__main__":
    main()