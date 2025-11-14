import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from datetime import datetime

class TrainingResultsAnalyzer:
    """训练结果分析器"""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.epoch_data = []
        self.metrics_history = {
            'train_loss': [], 'val_loss': [], 'learning_rate': [],
            'stance_accuracy': [], 'stance_f1': [],
            'harmfulness_accuracy': [], 'harmfulness_f1': [],
            'fairness_accuracy': [], 'fairness_f1': [],
            'intent_exact_match': [], 'intent_macro_f1': [],
            'intent_Political_f1': [], 'intent_Economic_f1': [],
            'intent_Psychological_f1': [], 'intent_Public_f1': []
        }
        
    def parse_log_file(self):
        """解析日志文件提取训练数据"""
        print("开始解析训练日志文件...")
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 调试：显示文件前500个字符
            print("文件前500字符:", content[:500])
            
            # 查找所有epoch完成的部分
            epoch_pattern = r'Epoch (\d+)/\d+\s*完成:(.*?)(?=\nEpoch \d+/\d+|\n===|\Z)'
            epochs = re.findall(epoch_pattern, content, re.DOTALL | re.IGNORECASE)
            
            print(f"找到 {len(epochs)} 个epoch的完成信息")
            
            for epoch_num, epoch_content in epochs:
                epoch_num = int(epoch_num)
                print(f"解析epoch {epoch_num}...")
                
                # 提取训练损失
                train_loss_match = re.search(r'训练损失:\s*([\d.]+)', epoch_content)
                val_loss_match = re.search(r'验证损失:\s*([\d.]+)', epoch_content)
                lr_match = re.search(r'当前学习率:\s*([\d.eE+-]+)', epoch_content)
                
                if train_loss_match and val_loss_match and lr_match:
                    epoch_info = {
                        'epoch': epoch_num,
                        'train_loss': float(train_loss_match.group(1)),
                        'val_loss': float(val_loss_match.group(1)),
                        'learning_rate': float(lr_match.group(1))
                    }
                    self.epoch_data.append(epoch_info)
                    print(f"  Epoch {epoch_num}: 训练损失={epoch_info['train_loss']:.4f}, "
                          f"验证损失={epoch_info['val_loss']:.4f}, LR={epoch_info['learning_rate']:.2e}")
                else:
                    print(f"  Epoch {epoch_num}: 缺失关键数据")
            
            # 提取验证指标
            self._parse_validation_metrics(content)
            
            # 按epoch排序
            self.epoch_data.sort(key=lambda x: x['epoch'])
            
            print(f"成功解析 {len(self.epoch_data)} 个epoch的训练数据")
            print(f"指标数据长度: {len(self.metrics_history['stance_accuracy'])}")
            
        except Exception as e:
            print(f"解析日志文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_validation_metrics(self, content):
        """解析验证指标"""
        print("开始解析验证指标...")
        
        # 查找所有验证指标部分
        val_pattern = r'验证指标:(.*?)(?=\n✓|\n⚠|\nEpoch \d+/\d+|\n=+|\Z)'
        val_sections = re.findall(val_pattern, content, re.DOTALL)
        
        print(f"找到 {len(val_sections)} 个验证指标部分")
        
        for i, section in enumerate(val_sections):
            # 清理section文本
            section = section.strip()
            if not section:
                continue
                
            print(f"解析第 {i+1} 个验证指标部分...")
            
            # 提取所有指标
            metrics = {}
            lines = section.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        # 尝试转换为浮点数
                        try:
                            metrics[key] = float(value)
                        except ValueError:
                            metrics[key] = 0.0
            
            # 更新指标历史
            for key in self.metrics_history.keys():
                if key in metrics:
                    self.metrics_history[key].append(metrics[key])
                    print(f"  {key}: {metrics[key]:.4f}")
                else:
                    self.metrics_history[key].append(0.0)
        
        print("验证指标解析完成!")
    
    def create_comprehensive_analysis(self):
        """创建全面的训练分析图表"""
        if not self.epoch_data:
            print("没有找到训练数据，请先调用parse_log_file()")
            return
        
        epochs = [data['epoch'] for data in self.epoch_data]
        
        print(f"开始创建分析图表，共有 {len(epochs)} 个epoch的数据")
        print(f"指标数据长度: {len(self.metrics_history['stance_accuracy'])}")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 损失函数和学习率
        ax1 = plt.subplot(3, 3, 1)
        self._plot_loss_and_lr(ax1, epochs)
        
        # 2. 主要任务准确率
        ax2 = plt.subplot(3, 3, 2)
        self._plot_main_accuracy(ax2, epochs)
        
        # 3. 主要任务F1分数
        ax3 = plt.subplot(3, 3, 3)
        self._plot_main_f1(ax3, epochs)
        
        # 4. Intent分类详细指标
        ax4 = plt.subplot(3, 3, 4)
        self._plot_intent_detailed(ax4, epochs)
        
        # 5. Intent各标签F1分数
        ax5 = plt.subplot(3, 3, 5)
        self._plot_intent_labels_f1(ax5, epochs)
        
        # 6. 损失函数对比分析
        ax6 = plt.subplot(3, 3, 6)
        self._plot_loss_comparison(ax6, epochs)
        
        # 7. 训练稳定性分析
        ax7 = plt.subplot(3, 3, 7)
        self._plot_training_stability(ax7, epochs)
        
        # 8. 性能指标热力图
        ax8 = plt.subplot(3, 3, 8)
        self._plot_performance_heatmap(ax8, epochs)
        
        # 9. 最终评估总结
        ax9 = plt.subplot(3, 3, 9)
        self._plot_final_summary(ax9)
        
        plt.tight_layout()
        plt.savefig('training_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 生成分析报告
        self._generate_analysis_report()
    
    def _plot_loss_and_lr(self, ax, epochs):
        """绘制损失函数和学习率"""
        train_loss = [data['train_loss'] for data in self.epoch_data]
        val_loss = [data['val_loss'] for data in self.epoch_data]
        learning_rates = [data['learning_rate'] for data in self.epoch_data]
        
        print(f"绘制损失函数: 训练损失{len(train_loss)}个点, 验证损失{len(val_loss)}个点")
        
        # 主Y轴：损失函数
        color = 'tab:red'
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color=color)
        line1 = ax.plot(epochs, train_loss, 'o-', color=color, label='Train Loss', linewidth=2, markersize=4)
        line2 = ax.plot(epochs, val_loss, 's-', color='tab:orange', label='Val Loss', linewidth=2, markersize=4)
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True, alpha=0.3)
        
        # 次Y轴：学习率
        ax2 = ax.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Learning Rate', color=color)
        line3 = ax2.plot(epochs, learning_rates, '^-', color=color, label='Learning Rate', alpha=0.7, markersize=4)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log')
        
        # 合并图例
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.set_title('训练损失、验证损失和学习率变化')
    
    def _plot_main_accuracy(self, ax, epochs):
        """绘制主要任务准确率"""
        metrics = ['stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy']
        labels = ['Stance', 'Harmfulness', 'Fairness']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for metric, label, color in zip(metrics, labels, colors):
            if len(self.metrics_history[metric]) >= len(epochs):
                values = self.metrics_history[metric][:len(epochs)]
                ax.plot(epochs, values, 'o-', label=label, color=color, linewidth=2, markersize=4)
                print(f"  {label}准确率: {len(values)}个点")
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('主要任务准确率变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 0.9)
    
    def _plot_main_f1(self, ax, epochs):
        """绘制主要任务F1分数"""
        metrics = ['stance_f1', 'harmfulness_f1', 'fairness_f1', 'intent_macro_f1']
        labels = ['Stance F1', 'Harmfulness F1', 'Fairness F1', 'Intent Macro F1']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for metric, label, color in zip(metrics, labels, colors):
            if len(self.metrics_history[metric]) >= len(epochs):
                values = self.metrics_history[metric][:len(epochs)]
                ax.plot(epochs, values, 's-', label=label, color=color, linewidth=2, markersize=4)
                print(f"  {label}: {len(values)}个点")
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('主要任务F1分数变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 0.9)
    
    def _plot_intent_detailed(self, ax, epochs):
        """绘制Intent分类详细指标"""
        if len(self.metrics_history['intent_exact_match']) >= len(epochs):
            exact_match = self.metrics_history['intent_exact_match'][:len(epochs)]
            macro_f1 = self.metrics_history['intent_macro_f1'][:len(epochs)]
            
            ax.plot(epochs, exact_match, 'o-', label='Exact Match', color='#9467bd', linewidth=2, markersize=4)
            ax.plot(epochs, macro_f1, 's-', label='Macro F1', color='#8c564b', linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Intent分类性能指标')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.1, 0.5)
    
    def _plot_intent_labels_f1(self, ax, epochs):
        """绘制Intent各标签F1分数"""
        intent_labels = ['Political', 'Economic', 'Psychological', 'Public']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, label in enumerate(intent_labels):
            metric = f'intent_{label}_f1'
            if len(self.metrics_history[metric]) >= len(epochs):
                values = self.metrics_history[metric][:len(epochs)]
                ax.plot(epochs, values, 'o-', label=label, color=colors[i], linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Intent各标签F1分数')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.9)
    
    def _plot_loss_comparison(self, ax, epochs):
        """绘制损失函数对比分析"""
        train_loss = [data['train_loss'] for data in self.epoch_data]
        val_loss = [data['val_loss'] for data in self.epoch_data]
        
        width = 0.35
        x = np.arange(len(epochs))
        
        ax.bar(x - width/2, train_loss, width, label='Train Loss', alpha=0.7, color='lightcoral')
        ax.bar(x + width/2, val_loss, width, label='Val Loss', alpha=0.7, color='lightblue')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('训练 vs 验证损失对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_stability(self, ax, epochs):
        """绘制训练稳定性分析"""
        if len(epochs) < 2:
            return
            
        # 计算各项指标的变化率
        metrics_to_analyze = ['train_loss', 'val_loss']
        labels = ['Train Loss Δ', 'Val Loss Δ']
        colors = ['#ff6b6b', '#4ecdc4']
        
        for i, metric in enumerate(metrics_to_analyze):
            if metric in ['train_loss', 'val_loss']:
                values = [data[metric] for data in self.epoch_data]
            else:
                continue
            
            # 计算变化率（百分比）
            changes = []
            for j in range(1, len(values)):
                if values[j-1] != 0:
                    change = ((values[j] - values[j-1]) / abs(values[j-1])) * 100
                    changes.append(change)
                else:
                    changes.append(0)
            
            if changes:
                ax.plot(epochs[1:], changes, 'o-', label=labels[i], color=colors[i], linewidth=1.5, alpha=0.8, markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Change Rate (%)')
        ax.set_title('训练稳定性分析（变化率）')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_performance_heatmap(self, ax, epochs):
        """绘制性能指标热力图"""
        if not self.epoch_data:
            return
            
        # 选择关键指标
        key_metrics = [
            'stance_accuracy', 'harmfulness_accuracy', 'fairness_accuracy',
            'intent_macro_f1', 'intent_Political_f1', 'intent_Public_f1'
        ]
        metric_names = ['Stance Acc', 'Harm Acc', 'Fair Acc', 'Intent F1', 'Political F1', 'Public F1']
        
        # 获取最终epoch的性能数据
        final_performance = []
        for metric in key_metrics:
            if self.metrics_history[metric]:
                final_performance.append(self.metrics_history[metric][-1] * 100)  # 转换为百分比
            else:
                final_performance.append(0)
        
        # 创建热力图数据
        performance_matrix = np.array(final_performance).reshape(1, -1)
        
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # 设置标签
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['Final\nPerformance'])
        
        # 添加数值标注
        for i, value in enumerate(final_performance):
            color = 'white' if value > 50 else 'black'
            ax.text(i, 0, f'{value:.1f}%', ha='center', va='center', color=color, fontweight='bold')
        
        ax.set_title('最终性能指标热力图')
        plt.colorbar(im, ax=ax, label='Performance (%)')
    
    def _plot_final_summary(self, ax):
        """绘制最终评估总结"""
        if not self.epoch_data:
            return
            
        # 获取最佳性能指标
        best_metrics = {}
        for metric in self.metrics_history:
            if self.metrics_history[metric]:
                best_metrics[metric] = max(self.metrics_history[metric])
        
        # 创建总结文本
        summary_text = []
        summary_text.append("训练结果总结")
        summary_text.append("="*30)
        summary_text.append(f"总训练轮次: {len(self.epoch_data)}")
        summary_text.append(f"最佳训练损失: {min([d['train_loss'] for d in self.epoch_data]):.4f}")
        summary_text.append(f"最佳验证损失: {min([d['val_loss'] for d in self.epoch_data]):.4f}")
        summary_text.append("")
        summary_text.append("最佳性能指标:")
        
        # 添加关键指标
        key_metrics = [
            ('stance_accuracy', 'Stance准确率'),
            ('harmfulness_accuracy', 'Harmfulness准确率'),
            ('fairness_accuracy', 'Fairness准确率'), 
            ('intent_macro_f1', 'Intent宏F1')
        ]
        
        for metric, name in key_metrics:
            if metric in best_metrics:
                summary_text.append(f"  {name}: {best_metrics[metric]:.3f}")
        
        # 显示文本
        ax.text(0.1, 0.95, '\n'.join(summary_text), transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('训练总结报告', fontsize=12, fontweight='bold')
    
    def _generate_analysis_report(self):
        """生成详细的分析报告"""
        print("\n" + "="*60)
        print("Thirteen版DMINT模型训练结果分析报告")
        print("="*60)
        
        if not self.epoch_data:
            print("无可用数据")
            return
        
        # 基本统计
        final_epoch = self.epoch_data[-1]
        print(f"\n📊 基本统计:")
        print(f"  总训练轮次: {len(self.epoch_data)}")
        print(f"  最终训练损失: {final_epoch['train_loss']:.4f}")
        print(f"  最终验证损失: {final_epoch['val_loss']:.4f}")
        print(f"  最终学习率: {final_epoch['learning_rate']:.2e}")
        
        # 性能分析
        print(f"\n🎯 性能分析:")
        metrics_to_report = [
            ('stance_accuracy', 'Stance准确率'),
            ('harmfulness_accuracy', 'Harmfulness准确率'), 
            ('fairness_accuracy', 'Fairness准确率'),
            ('intent_macro_f1', 'Intent宏F1')
        ]
        
        for metric, name in metrics_to_report:
            if self.metrics_history[metric]:
                final_value = self.metrics_history[metric][-1] if len(self.metrics_history[metric]) > 0 else 0
                best_value = max(self.metrics_history[metric]) if self.metrics_history[metric] else 0
                print(f"  {name}: {final_value:.3f} (最佳: {best_value:.3f})")
        
        # Intent标签分析
        print(f"\n🔍 Intent标签分析:")
        intent_labels = ['Political', 'Economic', 'Psychological', 'Public']
        for label in intent_labels:
            metric = f'intent_{label}_f1'
            if self.metrics_history[metric]:
                final_f1 = self.metrics_history[metric][-1] if len(self.metrics_history[metric]) > 0 else 0
                status = "✅ 良好" if final_f1 > 0.5 else "⚠️ 需要改进" if final_f1 > 0 else "❌ 严重问题"
                print(f"  {label}: {final_f1:.3f} {status}")
        
        # 训练建议
        print(f"\n💡 训练建议:")
        
        # 检查Psychological和Economic意图
        psych_f1 = self.metrics_history['intent_Psychological_f1'][-1] if self.metrics_history['intent_Psychological_f1'] else 0
        economic_f1 = self.metrics_history['intent_Economic_f1'][-1] if self.metrics_history['intent_Economic_f1'] else 0
        
        if psych_f1 == 0:
            print("  1. Psychological_fulfillment识别完全失败，需要专门优化")
        if economic_f1 == 0:
            print("  2. Economic_interest识别完全失败，需要重新设计特征")
        
        # 检查过拟合
        final_train_loss = final_epoch['train_loss']
        final_val_loss = final_epoch['val_loss']
        if final_train_loss > 0:
            overfitting_ratio = final_val_loss / final_train_loss
            if overfitting_ratio > 1.3:
                print(f"  3. 存在过拟合现象 (验证损失/训练损失 = {overfitting_ratio:.2f})")
        
        print(f"\n📁 分析图表已保存至: training_analysis_comprehensive.png")
        print("="*60)

# 使用示例
if __name__ == "__main__":
    # 创建分析器并解析日志
    analyzer = TrainingResultsAnalyzer("outputs\logs-34.txt")  # 替换为您的日志文件路径
    analyzer.parse_log_file()
    
    # 生成全面分析
    if analyzer.epoch_data:
        analyzer.create_comprehensive_analysis()
    else:
        print("无法解析训练数据，请检查日志文件格式")