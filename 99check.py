import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def comprehensive_data_analysis():
    """全面的数据探索分析"""
    print("开始数据探索分析...")
    
    # 加载数据
    data_dir = "data/"
    
    try:
        # 读取CSV文件
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), sep=',', header=None,
                              names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
        val_df = pd.read_csv(os.path.join(data_dir, "val.csv"), sep=',', header=None,
                            names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
        test_df = pd.read_csv(os.path.join(data_dir, "test1.csv"), sep=',', header=None,
                             names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
        
        print("✓ CSV文件加载成功")
    except Exception as e:
        print(f"✗ CSV文件加载失败: {e}")
        return None, None, None
    
    # 加载JSON数据并整合文本
    def load_json_data():
        topics_path = os.path.join(data_dir, "news_topic1.json")
        docs_path = os.path.join(data_dir, "news_docs.json")
        
        topics_data = {}
        docs_data = {}
        
        try:
            if os.path.exists(topics_path):
                with open(topics_path, 'r', encoding='utf-8') as f:
                    topics_data = json.load(f)
                print("✓ 主题数据加载成功")
            
            if os.path.exists(docs_path):
                with open(docs_path, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                print("✓ 文档数据加载成功")
        except Exception as e:
            print(f"✗ JSON文件加载失败: {e}")
        
        return topics_data, docs_data
    
    topics_data, docs_data = load_json_data()
    
    def get_text_content(row_id):
        """获取文本内容"""
        str_id = str(row_id)
        topic_text = topics_data.get(str_id, "unknown topic")
        
        doc_content = ""
        if docs_data and str_id in docs_data:
            doc_content = docs_data.get(str_id, {}).get("content", "")
        
        if not doc_content.strip():
            return topic_text
        else:
            return f"{topic_text} {doc_content}"
    
    # 为数据框添加文本列
    train_df['text'] = train_df['id'].apply(get_text_content)
    val_df['text'] = val_df['id'].apply(get_text_content)
    test_df['text'] = test_df['id'].apply(get_text_content)
    
    print("=" * 80)
    print("数据探索分析报告")
    print("=" * 80)
    
    # 1. 基础统计
    print("\n1. 数据集基础统计:")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    # 2. 标签分布分析
    print("\n2. 标签分布分析:")
    analyze_label_distribution(train_df, "训练集")
    analyze_label_distribution(val_df, "验证集")
    analyze_label_distribution(test_df, "测试集")
    
    # 3. 意图标签详细分析
    print("\n3. 意图标签详细分析:")
    analyze_intent_distribution(train_df, "训练集")
    analyze_intent_distribution(val_df, "验证集")
    analyze_intent_distribution(test_df, "测试集")
    
    # 4. 文本分析
    print("\n4. 文本特征分析:")
    analyze_text_features(train_df, "训练集")
    analyze_text_features(val_df, "验证集")
    analyze_text_features(test_df, "测试集")
    
    # 5. 标签相关性分析
    print("\n5. 标签相关性分析:")
    analyze_label_correlations(train_df, "训练集")
    
    # 6. 数据质量问题检测
    print("\n6. 数据质量问题检测:")
    detect_data_quality_issues(train_df, "训练集")
    detect_data_quality_issues(val_df, "验证集")
    detect_data_quality_issues(test_df, "测试集")
    
    # 7. 可视化分析
    print("\n7. 生成可视化图表...")
    generate_visualizations(train_df, val_df, test_df)
    
    return train_df, val_df, test_df

def analyze_label_distribution(df, dataset_name):
    """分析各任务标签分布"""
    print(f"\n{dataset_name}标签分布:")
    
    # Stance分布
    print("  Stance立场分布:")
    stance_dist = df['stance'].value_counts().sort_index()
    stance_map = {'Against': 0, 'Neither': 1, 'Favor': 2, '0': 0, '1': 1, '2': 2}
    
    # 统一标签格式
    stance_numeric = []
    for stance in df['stance']:
        if stance in stance_map:
            stance_numeric.append(stance_map[stance])
        else:
            stance_numeric.append(1)  # 默认Neither
    
    stance_counts = Counter(stance_numeric)
    total = len(df)
    
    for stance_code in [0, 1, 2]:
        count = stance_counts.get(stance_code, 0)
        percentage = (count / total) * 100
        label_name = {0: 'Against', 1: 'Neither', 2: 'Favor'}[stance_code]
        print(f"    {label_name}: {count} ({percentage:.1f}%)")
    
    # Fairness分布
    print("  Fairness公平性分布:")
    fairness_map = {'Tinted': 0, 'Fairness': 1, '0': 0, '1': 1}
    fairness_numeric = []
    for fairness in df['fairness']:
        if fairness in fairness_map:
            fairness_numeric.append(fairness_map[fairness])
        else:
            fairness_numeric.append(0)  # 默认Tinted
    
    fairness_counts = Counter(fairness_numeric)
    for fairness_code in [0, 1]:
        count = fairness_counts.get(fairness_code, 0)
        percentage = (count / total) * 100
        label_name = {0: 'Tinted', 1: 'Fairness'}[fairness_code]
        print(f"    {label_name}: {count} ({percentage:.1f}%)")
    
    # Harmfulness分布
    print("  Harmfulness危害性分布:")
    harmfulness_map = {'Harmful': 0, 'Unharmful': 1, '0': 0, '1': 1}
    harmfulness_numeric = []
    for harmfulness in df['harmfulness']:
        if harmfulness in harmfulness_map:
            harmfulness_numeric.append(harmfulness_map[harmfulness])
        else:
            harmfulness_numeric.append(0)  # 默认Harmful
    
    harmfulness_counts = Counter(harmfulness_numeric)
    for harmfulness_code in [0, 1]:
        count = harmfulness_counts.get(harmfulness_code, 0)
        percentage = (count / total) * 100
        label_name = {0: 'Harmful', 1: 'Unharmful'}[harmfulness_code]
        print(f"    {label_name}: {count} ({percentage:.1f}%)")

def analyze_intent_distribution(df, dataset_name):
    """详细分析意图标签分布"""
    intent_labels = ['Political_interest', 'Economic_interest', 
                    'Psychological_fulfillment', 'Public_interest']
    
    print(f"\n{dataset_name}意图分布:")
    
    # 单个标签计数
    single_counts = {label: 0 for label in intent_labels}
    total_samples = len(df)
    valid_intent_samples = 0
    
    for idx, row in df.iterrows():
        intent_str = str(row['intent'])
        if pd.isna(intent_str) or intent_str == 'nan' or intent_str == 'None':
            continue
            
        valid_intent_samples += 1
        intent_list = intent_str.split('&')
        for label in intent_labels:
            if label in intent_list:
                single_counts[label] += 1
    
    # 打印单个标签分布
    print("  单个意图标签分布:")
    for label in intent_labels:
        if total_samples > 0:
            percentage = (single_counts[label] / total_samples) * 100
        else:
            percentage = 0
        print(f"    {label}: {single_counts[label]} ({percentage:.1f}%)")
    
    # 分析多标签组合
    print(f"\n  {dataset_name}多标签组合分析:")
    label_combinations = {}
    for idx, row in df.iterrows():
        intent_str = str(row['intent'])
        if pd.isna(intent_str) or intent_str == 'nan' or intent_str == 'None':
            continue
            
        intent_list = intent_str.split('&')
        # 过滤有效标签
        valid_intents = [intent for intent in intent_list if intent in intent_labels]
        if valid_intents:
            key = '&'.join(sorted(valid_intents))
            label_combinations[key] = label_combinations.get(key, 0) + 1
    
    # 显示最常见的组合
    if label_combinations:
        top_combinations = sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)[:10]
        for combo, count in top_combinations:
            percentage = (count / total_samples) * 100
            print(f"    {combo}: {count} ({percentage:.1f}%)")
    else:
        print("    无有效的多标签组合")
    
    # 标签数量分布
    print(f"\n  {dataset_name}每个样本的意图标签数量:")
    label_counts = []
    for idx, row in df.iterrows():
        intent_str = str(row['intent'])
        if pd.isna(intent_str) or intent_str == 'nan' or intent_str == 'None':
            label_counts.append(0)
        else:
            intent_list = intent_str.split('&')
            valid_intents = [intent for intent in intent_list if intent in intent_labels]
            label_counts.append(len(valid_intents))
    
    if label_counts:
        count_dist = Counter(label_counts)
        for count in sorted(count_dist.keys()):
            freq = count_dist[count]
            percentage = (freq / total_samples) * 100
            print(f"    {count}个标签: {freq} ({percentage:.1f}%)")
    else:
        print("    无有效的意图标签")
    
    print(f"  有效意图样本数: {valid_intent_samples}/{total_samples} ({valid_intent_samples/total_samples*100:.1f}%)")

def analyze_text_features(df, dataset_name):
    """分析文本特征"""
    print(f"\n{dataset_name}文本特征:")
    
    # 文本长度分析
    text_lengths = df['text'].str.len()
    print(f"  文本长度(字符) - 平均: {text_lengths.mean():.1f}, 最小: {text_lengths.min()}, 最大: {text_lengths.max()}")
    
    # 分词长度分析（估算）
    word_counts = df['text'].str.split().str.len()
    print(f"  词汇数量 - 平均: {word_counts.mean():.1f}, 最小: {word_counts.min()}, 最大: {word_counts.max()}")
    
    # 检查空文本
    empty_texts = (text_lengths == 0).sum()
    print(f"  空文本数量: {empty_texts}")
    
    # 文本重复分析
    duplicate_texts = df['text'].duplicated().sum()
    print(f"  重复文本数量: {duplicate_texts}")
    
    # 检查文本唯一性
    unique_texts = df['text'].nunique()
    print(f"  唯一文本数量: {unique_texts}/{len(df)} ({unique_texts/len(df)*100:.1f}%)")

def analyze_label_correlations(df, dataset_name):
    """分析标签间的相关性"""
    print(f"\n{dataset_name}标签间相关性分析:")
    
    # 创建数值化的标签DataFrame用于相关性分析
    analysis_df = pd.DataFrame()
    
    # Stance
    stance_map = {'Against': 0, 'Neither': 1, 'Favor': 2, '0': 0, '1': 1, '2': 2}
    analysis_df['stance'] = df['stance'].apply(lambda x: stance_map.get(x, 1))
    
    # Fairness
    fairness_map = {'Tinted': 0, 'Fairness': 1, '0': 0, '1': 1}
    analysis_df['fairness'] = df['fairness'].apply(lambda x: fairness_map.get(x, 0))
    
    # Harmfulness  
    harmfulness_map = {'Harmful': 0, 'Unharmful': 1, '0': 0, '1': 1}
    analysis_df['harmfulness'] = df['harmfulness'].apply(lambda x: harmfulness_map.get(x, 0))
    
    # Intent标签
    intent_labels = ['Political_interest', 'Economic_interest', 
                    'Psychological_fulfillment', 'Public_interest']
    
    for label in intent_labels:
        analysis_df[f'intent_{label}'] = df['intent'].apply(
            lambda x: 1 if pd.notna(x) and label in str(x) else 0
        )
    
    # 计算相关性矩阵
    correlation_matrix = analysis_df.corr()
    
    print("  主要相关性 (|corr| > 0.2):")
    found_correlations = False
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.2:
                print(f"    {correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {corr:.3f}")
                found_correlations = True
    
    if not found_correlations:
        print("    未发现显著相关性 (|corr| > 0.2)")

def detect_data_quality_issues(df, dataset_name):
    """检测数据质量问题"""
    print(f"\n{dataset_name}数据质量问题检测:")
    
    issues = []
    
    # 1. 检查缺失值
    missing_data = df[['stance', 'intent', 'fairness', 'harmfulness', 'text']].isnull().sum()
    for col, count in missing_data.items():
        if count > 0:
            issues.append(f"    {col}有{count}个缺失值")
    
    # 2. 检查文本质量问题
    empty_texts = (df['text'].str.len() == 0).sum()
    if empty_texts > 0:
        issues.append(f"    有{empty_texts}个空文本")
    
    # 3. 检查标签格式问题
    valid_stance = ['Against', 'Neither', 'Favor', '0', '1', '2']
    invalid_stance = df[~df['stance'].isin(valid_stance)]['stance'].unique()
    if len(invalid_stance) > 0:
        issues.append(f"    stance有无效值: {invalid_stance}")
    
    # 4. 检查意图标签格式
    intent_labels = ['Political_interest', 'Economic_interest', 
                    'Psychological_fulfillment', 'Public_interest']
    invalid_intent_count = 0
    for intent in df['intent']:
        if pd.isna(intent) or intent == 'nan' or intent == 'None':
            continue
        intent_list = str(intent).split('&')
        for item in intent_list:
            if item not in intent_labels and item.strip():
                invalid_intent_count += 1
                break
    
    if invalid_intent_count > 0:
        issues.append(f"    intent有{invalid_intent_count}个样本包含无效标签")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("    未发现明显数据质量问题")

def generate_visualizations(train_df, val_df, test_df):
    """生成可视化图表"""
    try:
        # 创建可视化目录
        os.makedirs('data_analysis', exist_ok=True)
        
        # 1. 标签分布对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('数据集标签分布对比', fontsize=16)
        
        # Stance分布
        stance_data = []
        for df, name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
            stance_map = {'Against': 0, 'Neither': 1, 'Favor': 2, '0': 0, '1': 1, '2': 2}
            stance_numeric = [stance_map.get(x, 1) for x in df['stance']]
            counts = Counter(stance_numeric)
            for stance_code in [0, 1, 2]:
                stance_data.append({
                    '数据集': name,
                    '立场': {0: 'Against', 1: 'Neither', 2: 'Favor'}[stance_code],
                    '数量': counts.get(stance_code, 0)
                })
        
        stance_df = pd.DataFrame(stance_data)
        pivot_stance = stance_df.pivot(index='立场', columns='数据集', values='数量')
        pivot_stance.plot(kind='bar', ax=axes[0,0], title='Stance立场分布')
        axes[0,0].set_ylabel('样本数量')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Intent分布
        intent_labels = ['Political_interest', 'Economic_interest', 
                        'Psychological_fulfillment', 'Public_interest']
        intent_data = []
        for df, name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
            for label in intent_labels:
                count = sum(1 for intent in df['intent'] if pd.notna(intent) and label in str(intent))
                intent_data.append({
                    '数据集': name,
                    '意图': label,
                    '数量': count
                })
        
        intent_df = pd.DataFrame(intent_data)
        pivot_intent = intent_df.pivot(index='意图', columns='数据集', values='数量')
        pivot_intent.plot(kind='bar', ax=axes[0,1], title='Intent意图分布')
        axes[0,1].set_ylabel('样本数量')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Fairness分布
        fairness_data = []
        for df, name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
            fairness_map = {'Tinted': 0, 'Fairness': 1, '0': 0, '1': 1}
            fairness_numeric = [fairness_map.get(x, 0) for x in df['fairness']]
            counts = Counter(fairness_numeric)
            for fairness_code in [0, 1]:
                fairness_data.append({
                    '数据集': name,
                    '公平性': {0: 'Tinted', 1: 'Fairness'}[fairness_code],
                    '数量': counts.get(fairness_code, 0)
                })
        
        fairness_df = pd.DataFrame(fairness_data)
        pivot_fairness = fairness_df.pivot(index='公平性', columns='数据集', values='数量')
        pivot_fairness.plot(kind='bar', ax=axes[1,0], title='Fairness公平性分布')
        axes[1,0].set_ylabel('样本数量')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Harmfulness分布
        harmfulness_data = []
        for df, name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
            harmfulness_map = {'Harmful': 0, 'Unharmful': 1, '0': 0, '1': 1}
            harmfulness_numeric = [harmfulness_map.get(x, 0) for x in df['harmfulness']]
            counts = Counter(harmfulness_numeric)
            for harmfulness_code in [0, 1]:
                harmfulness_data.append({
                    '数据集': name,
                    '危害性': {0: 'Harmful', 1: 'Unharmful'}[harmfulness_code],
                    '数量': counts.get(harmfulness_code, 0)
                })
        
        harmfulness_df = pd.DataFrame(harmfulness_data)
        pivot_harmfulness = harmfulness_df.pivot(index='危害性', columns='数据集', values='数量')
        pivot_harmfulness.plot(kind='bar', ax=axes[1,1], title='Harmfulness危害性分布')
        axes[1,1].set_ylabel('样本数量')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_analysis/label_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 文本长度分布
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('文本长度分布', fontsize=16)
        
        for i, (df, name) in enumerate([(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]):
            text_lengths = df['text'].str.len()
            axes[i].hist(text_lengths, bins=50, alpha=0.7)
            axes[i].set_title(f'{name}文本长度分布')
            axes[i].set_xlabel('文本长度(字符)')
            axes[i].set_ylabel('频次')
            axes[i].axvline(text_lengths.mean(), color='red', linestyle='--', label=f'平均: {text_lengths.mean():.1f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('data_analysis/text_length_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 可视化图表已保存至 data_analysis/ 目录")
        
    except Exception as e:
        print(f"✗ 可视化生成失败: {e}")

if __name__ == "__main__":
    train_df, val_df, test_df = comprehensive_data_analysis()
    
    if train_df is not None:
        print("\n" + "="*80)
        print("数据探索完成！请查看上面的分析报告和生成的图表。")
        print("="*80)
        
        # 保存处理后的数据
        try:
            train_df.to_csv('data_analysis/train_analyzed.csv', index=False, encoding='utf-8')
            val_df.to_csv('data_analysis/val_analyzed.csv', index=False, encoding='utf-8')
            test_df.to_csv('data_analysis/test_analyzed.csv', index=False, encoding='utf-8')
            print("✓ 处理后的数据已保存至 data_analysis/ 目录")
        except Exception as e:
            print(f"✗ 数据保存失败: {e}")
    else:
        print("\n数据探索失败，请检查数据路径和格式。")
        