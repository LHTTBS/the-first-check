"""
Fourteen.py - 增强版DMINT模型，融合评论数据
在Thirteen版基础上增加评论数据，从数据种类层面提高训练效果
"""

import pandas as pd
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from typing import Dict, List, Tuple
import time
from collections import Counter, deque
import math

# 设置模型路径
MODEL_PATH = "models\chinese-roberta-wwm-ext"

class WarmupPolyLR:
    """Warmup + 多项式衰减学习率调度器"""
    
    def __init__(self, optimizer, total_epochs, warmup_epochs=2, poly_exp=0.9, min_lr=1e-6, base_lr=2e-5):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.poly_exp = poly_exp
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0
        
    def step(self):
        """更新学习率"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup阶段：线性增加
            lr_scale = self.current_epoch / self.warmup_epochs
            new_lr = self.base_lr * lr_scale
        else:
            # 多项式衰减
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = (1.0 - progress) ** self.poly_exp
            new_lr = max(self.base_lr * lr_scale, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        return new_lr

class EnhancedTextDataset(Dataset):
    """增强版文本数据集，添加评论数据和平衡处理"""
    
    def __init__(self, dataframe, tokenizer, max_length=256, phase="train"):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.phase = phase
        
        # 标签映射
        self.stance_map = {'Against': 0, 'Neither': 1, 'Favor': 2}
        self.fairness_map = {'Tinted': 0, 'Fairness': 1}
        self.harmfulness_map = {'Harmful': 0, 'Unharmful': 1}
        
        # Intent是多标签分类
        self.intent_labels = ['Political_interest', 'Economic_interest', 
                             'Psychological_fulfillment', 'Public_interest']
        
        # 加载评论数据
        self.id_post_mapping, self.post_data = self.load_comment_data()
        
        # 分析数据分布
        self.analyze_data_distribution()
        
    def load_comment_data(self):
        """加载评论相关数据"""
        data_dir = "data/"
        
        # 加载id_post_pair映射
        id_post_path = os.path.join(data_dir, "id_post_pair.json")
        id_post_mapping = {}
        if os.path.exists(id_post_path):
            with open(id_post_path, 'r', encoding='utf-8') as f:
                id_post_mapping = json.load(f)
            print(f"加载了 {len(id_post_mapping)} 个ID到帖子的映射")
        
        # 加载帖子数据
        post_path = os.path.join(data_dir, "post.json")
        post_data = {}
        if os.path.exists(post_path):
            with open(post_path, 'r', encoding='utf-8') as f:
                posts = json.load(f)
                for post in posts:
                    post_data[post['pid']] = post
            print(f"加载了 {len(post_data)} 个帖子数据")
        
        # 加载帖子文档数据（备用）
        post_docs_path = os.path.join(data_dir, "post_docs.json")
        post_docs_data = {}
        if os.path.exists(post_docs_path):
            with open(post_docs_path, 'r', encoding='utf-8') as f:
                post_docs = json.load(f)
                for doc in post_docs:
                    post_docs_data[doc['post_id']] = doc
            print(f"加载了 {len(post_docs_data)} 个帖子文档数据")
        
        return id_post_mapping, post_data
    
    def get_comment_tree_text(self, post_id):
        """获取帖子及其评论树的完整文本"""
        if post_id not in self.post_data:
            return ""
        
        # 使用BFS遍历评论树
        all_texts = []
        queue = deque([post_id])
        
        while queue:
            current_pid = queue.popleft()
            if current_pid in self.post_data:
                post = self.post_data[current_pid]
                # 添加当前帖子/评论的文本
                if 'content' in post and post['content']:
                    all_texts.append(post['content'])
                
                # 添加子评论
                if 'child' in post:
                    for child_pid in post['child']:
                        queue.append(child_pid)
        
        return " ".join(all_texts)
    
    def get_enhanced_text(self, row_id, original_text):
        """获取增强的文本（原始文本 + 评论数据）"""
        str_id = str(row_id)
        
        # 如果ID在映射中，获取对应的帖子ID
        if str_id in self.id_post_mapping:
            post_id = self.id_post_mapping[str_id]
            comment_text = self.get_comment_tree_text(post_id)
            
            if comment_text:
                # 合并原始文本和评论文本
                enhanced_text = f"{original_text} [评论上下文] {comment_text}"
                return enhanced_text[:5000]  # 限制总长度
        
        return original_text
    
    def analyze_data_distribution(self):
        """分析数据分布，识别问题"""
        print(f"\n=== {self.phase}集数据分布分析 ===")
        
        # Stance分布
        stance_counts = Counter(self.data['stance'])
        print(f"Stance分布: {dict(stance_counts)}")
        
        # Intent分布分析
        intent_label_counts = [0] * len(self.intent_labels)
        intent_combinations = Counter()
        
        for idx, row in self.data.iterrows():
            intent_str = str(row['intent'])
            if pd.isna(intent_str):
                continue
                
            intent_list = intent_str.split('&')
            intent_combinations[intent_str] += 1
            
            for i, label in enumerate(self.intent_labels):
                if label in intent_list:
                    intent_label_counts[i] += 1
        
        print(f"Intent标签分布: {dict(zip(self.intent_labels, intent_label_counts))}")
        print(f"前5个Intent组合: {intent_combinations.most_common(5)}")
        
        # 特别关注Psychological_fulfillment
        psych_idx = self.intent_labels.index('Psychological_fulfillment')
        psych_count = intent_label_counts[psych_idx]
        print(f"Psychological_fulfillment样本数: {psych_count} (占比: {psych_count/len(self.data)*100:.2f}%)")
        
        # Fairness分布
        fairness_counts = Counter(self.data['fairness'])
        print(f"Fairness分布: {dict(fairness_counts)}")
        
        # Harmfulness分布
        harmfulness_counts = Counter(self.data['harmfulness'])
        print(f"Harmfulness分布: {dict(harmfulness_counts)}")
        
        # 分析评论数据覆盖情况
        comment_coverage = 0
        for idx, row in self.data.iterrows():
            str_id = str(row['id'])
            if str_id in self.id_post_mapping:
                comment_coverage += 1
        
        print(f"评论数据覆盖: {comment_coverage}/{len(self.data)} ({comment_coverage/len(self.data)*100:.2f}%)")
    
    def __len__(self):
        return len(self.data)
    
    def encode_intent(self, intent_str):
        """将多标签intent编码为二进制向量"""
        if pd.isna(intent_str):
            return [0] * len(self.intent_labels)
        
        intent_list = str(intent_str).split('&')
        encoding = [0] * len(self.intent_labels)
        
        for i, label in enumerate(self.intent_labels):
            if label in intent_list:
                encoding[i] = 1
                
        return encoding
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 获取基础文本
        base_text = str(row['text']) if 'text' in row else "default text"
        
        # 获取增强文本（包含评论）
        enhanced_text = self.get_enhanced_text(row['id'], base_text)
        
        # 编码文本
        encoding = self.tokenizer(
            enhanced_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 获取标签 - 添加容错处理
        stance = self.stance_map.get(str(row['stance']), 1)  # 默认Neither
        fairness = self.fairness_map.get(str(row['fairness']), 0)  # 默认Tinted
        harmfulness = self.harmfulness_map.get(str(row['harmfulness']), 0)  # 默认Harmful
        intent = self.encode_intent(row['intent'])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'stance': torch.tensor(stance, dtype=torch.long),
            'intent': torch.tensor(intent, dtype=torch.float),
            'fairness': torch.tensor(fairness, dtype=torch.long),
            'harmfulness': torch.tensor(harmfulness, dtype=torch.long),
            'text_length': torch.tensor(len(enhanced_text), dtype=torch.long)
        }

class OptimizedMultiScaleFeatureExtractor(nn.Module):
    """优化版多尺度特征提取器"""
    
    def __init__(self, input_dim: int = 768, output_dim: int = 256, dropout: float = 0.1):
        super(OptimizedMultiScaleFeatureExtractor, self).__init__()
        
        # 多尺度卷积层
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1, padding=0)   # 单词语义
        self.conv2 = nn.Conv1d(input_dim, 64, kernel_size=2, padding=1)   # 二元关系
        self.conv3 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)   # 三元短语
        self.conv4 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)   # 短句模式
        self.conv5 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)   # 长距离依赖
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv1d(input_dim, 320, kernel_size=1),
            nn.BatchNorm1d(320)
        )
        
        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Linear(320, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转置为卷积需要的格式 [batch_size, hidden_size, seq_len]
        x_input = x.transpose(1, 2)
        
        # 多尺度卷积
        feat1 = F.relu(self.conv1(x_input))
        feat2 = F.relu(self.conv2(x_input))
        feat3 = F.relu(self.conv3(x_input))
        feat4 = F.relu(self.conv4(x_input))
        feat5 = F.relu(self.conv5(x_input))
        
        # 全局平均池化
        pool1 = F.adaptive_avg_pool1d(feat1, 1).squeeze(-1)
        pool2 = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)
        pool3 = F.adaptive_avg_pool1d(feat3, 1).squeeze(-1)
        pool4 = F.adaptive_avg_pool1d(feat4, 1).squeeze(-1)
        pool5 = F.adaptive_avg_pool1d(feat5, 1).squeeze(-1)
        
        # 拼接多尺度特征 + 残差连接
        multi_scale_features = torch.cat([pool1, pool2, pool3, pool4, pool5], dim=1)
        residual_features = F.adaptive_avg_pool1d(self.residual(x_input), 1).squeeze(-1)
        
        # 特征融合
        fused_features = self.fusion(multi_scale_features + residual_features)
        
        return fused_features

class EnhancedDifferentiatedFeatureExtractor(nn.Module):
    """增强版差异化特征提取器"""
    
    def __init__(self, perspective_type: str, input_dim: int = 768, output_dim: int = 256, dropout: float = 0.1):
        super(EnhancedDifferentiatedFeatureExtractor, self).__init__()
        self.perspective_type = perspective_type
        
        if perspective_type == "belief":
            # 信念视角：更深网络 + 自注意力
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 384),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(384, output_dim),
            )
            self.attention = nn.MultiheadAttention(output_dim, 4, dropout=dropout)
            
        elif perspective_type == "desire":
            # 欲望视角：中等深度 + 通道注意力
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 448),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(448, output_dim),
            )
            self.channel_attention = nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim),
                nn.Sigmoid()
            )
        else:  # plan
            # 计划视角：更浅网络 + 残差连接
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 320),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(320, output_dim),
            )
            self.residual = nn.Linear(input_dim, output_dim)
        
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.perspective_type == "belief":
            features = self.feature_net(x)
            # 自注意力增强
            features_attn = features.unsqueeze(1)
            attended, _ = self.attention(features_attn, features_attn, features_attn)
            features = features + attended.squeeze(1)
            
        elif self.perspective_type == "desire":
            features = self.feature_net(x)
            # 通道注意力
            attention_weights = self.channel_attention(features)
            features = features * attention_weights
            
        else:  # plan
            base_features = self.feature_net(x)
            residual_features = self.residual(x)
            features = base_features + residual_features
        
        features = self.layer_norm(features)
        return self.dropout(features)

class CorrelationEnhancedIntentClassifier(nn.Module):
    """相关性增强的意图分类器 - 专门解决Psychological_fulfillment问题"""
    
    def __init__(self, feature_dim: int = 256, num_intents: int = 4, dropout: float = 0.2):
        super(CorrelationEnhancedIntentClassifier, self).__init__()
        
        # 为Psychological_fulfillment创建专用路径
        self.psych_special_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # 更少的dropout
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )
        
        # 其他意图的共享网络
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_intents - 1)  # 除了Psychological_fulfillment
        )
        
        # 标签相关性矩阵 - 可学习参数
        self.label_correlation = nn.Parameter(torch.eye(num_intents))
        
        # 注意力机制捕捉标签间关系
        self.attention = nn.MultiheadAttention(
            embed_dim=num_intents, 
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        
        self.num_intents = num_intents
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Psychological_fulfillment专用预测
        psych_logit = self.psych_special_head(x)  # [batch_size, 1]
        
        # 其他意图的共享预测
        other_logits = self.shared_net(x)  # [batch_size, num_intents-1]
        
        # 合并所有logits
        intent_logits = torch.cat([other_logits[:, :1], psych_logit, other_logits[:, 1:]], dim=1)
        
        # 应用标签相关性矩阵
        correlated_logits = torch.matmul(intent_logits, self.label_correlation)
        
        # 使用注意力机制进一步建模标签关系
        logits_reshaped = correlated_logits.unsqueeze(1)  # [batch_size, 1, num_intents]
        attended_logits, _ = self.attention(logits_reshaped, logits_reshaped, logits_reshaped)
        attended_logits = attended_logits.squeeze(1)
        
        # 残差连接
        final_logits = correlated_logits + 0.3 * attended_logits
        
        return final_logits

class EnhancedFairnessClassifier(nn.Module):
    """增强版公平性分类器"""
    
    def __init__(self, feature_dim: int = 256, dropout: float = 0.2):
        super(EnhancedFairnessClassifier, self).__init__()
        
        # 更深的网络结构来捕捉公平性特征
        self.fairness_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Tinted, Fairness
        )
        
        # 注意力机制聚焦关键特征
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用注意力机制增强特征
        x_reshaped = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        attended_x, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attended_x = attended_x.squeeze(1)
        
        # 残差连接
        enhanced_features = x + 0.2 * attended_x
        
        # 分类
        fairness_logits = self.fairness_net(enhanced_features)
        
        return fairness_logits

class AdvancedGatedIntentAggregator(nn.Module):
    """高级门控意图聚合器"""
    
    def __init__(self, feature_dim: int = 256, num_intents: int = 4, dropout: float = 0.1):
        super(AdvancedGatedIntentAggregator, self).__init__()
        
        # 多层次门控机制
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 3),  # 3个视角的权重
            nn.Softmax(dim=1)
        )
        
        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 使用相关性增强的意图分类器
        self.intent_classifier = CorrelationEnhancedIntentClassifier(feature_dim, num_intents, dropout)
        
    def forward(self, belief_features: torch.Tensor, desire_features: torch.Tensor, 
                plan_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # 拼接所有视角特征
        concatenated = torch.cat([belief_features, desire_features, plan_features], dim=1)
        
        # 计算门控权重
        gate_weights = self.gate_network(concatenated)  # [batch_size, 3]
        
        # 应用门控权重的加权融合
        weighted_belief = belief_features * gate_weights[:, 0:1]
        weighted_desire = desire_features * gate_weights[:, 1:2]
        weighted_plan = plan_features * gate_weights[:, 2:3]
        
        # 拼接加权特征
        weighted_concatenated = torch.cat([weighted_belief, weighted_desire, weighted_plan], dim=1)
        
        # 特征融合
        fused_features = self.fusion_network(weighted_concatenated)
        
        # 意图分类
        intent_logits = self.intent_classifier(fused_features)
        
        return {
            'intent_logits': intent_logits,
            'fused_features': fused_features,
            'gate_weights': gate_weights
        }

class FourteenDMINTModel(nn.Module):
    """Fourteen版DMINT模型 - 融合评论数据和所有优化"""
    
    def __init__(self, bert_model, hidden_size: int = 768, feature_dim: int = 256, dropout: float = 0.3):
        super(FourteenDMINTModel, self).__init__()
        
        self.bert = bert_model
        
        # 使用优化版多尺度特征提取器
        self.multi_scale_extractor = OptimizedMultiScaleFeatureExtractor(hidden_size, feature_dim, dropout)
        
        # 使用增强版差异化特征提取器
        self.belief_extractor = EnhancedDifferentiatedFeatureExtractor("belief", hidden_size, feature_dim, dropout)
        self.desire_extractor = EnhancedDifferentiatedFeatureExtractor("desire", hidden_size, feature_dim, dropout)
        self.plan_extractor = EnhancedDifferentiatedFeatureExtractor("plan", hidden_size, feature_dim, dropout)
        
        # 使用高级门控意图聚合器
        self.intent_aggregator = AdvancedGatedIntentAggregator(feature_dim, 4, dropout)
        
        # 任务特定的分类器
        self.stance_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # Against, Neither, Favor
        )
        
        self.harmfulness_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Harmful, Unharmful
        )
        
        # 使用增强版公平性分类器
        self.fairness_classifier = EnhancedFairnessClassifier(feature_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # BERT特征提取
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        cls_features = bert_output.pooler_output
        
        # 多尺度特征提取（基于序列输出）
        multi_scale_features = self.multi_scale_extractor(sequence_output)
        
        # 三个视角的特征提取（基于CLS特征）
        belief_features = self.belief_extractor(cls_features)
        desire_features = self.desire_extractor(cls_features)
        plan_features = self.plan_extractor(cls_features)
        
        # 应用dropout
        belief_features = self.dropout(belief_features)
        desire_features = self.dropout(desire_features)
        plan_features = self.dropout(plan_features)
        multi_scale_features = self.dropout(multi_scale_features)
        
        # 门控意图聚合
        intent_output = self.intent_aggregator(belief_features, desire_features, plan_features)
        fused_features = intent_output['fused_features']
        intent_logits = intent_output['intent_logits']
        
        # 任务分类 - 结合多尺度特征和视角特征
        stance_input = torch.cat([multi_scale_features, belief_features], dim=1)
        stance_logits = self.stance_classifier(stance_input)
        
        harmfulness_input = torch.cat([multi_scale_features, plan_features], dim=1)
        harmfulness_logits = self.harmfulness_classifier(harmfulness_input)
        
        fairness_logits = self.fairness_classifier(fused_features)
        
        return {
            'stance': stance_logits,
            'intent': intent_logits,
            'harmfulness': harmfulness_logits,
            'fairness': fairness_logits,
            'belief_features': belief_features,
            'desire_features': desire_features,
            'plan_features': plan_features,
            'multi_scale_features': multi_scale_features,
            'gate_weights': intent_output['gate_weights']
        }

def compute_enhanced_class_weights(train_df):
    """计算增强的类别权重，特别优化Psychological_fulfillment"""
    intent_labels = ['Political_interest', 'Economic_interest', 
                    'Psychological_fulfillment', 'Public_interest']
    
    # 统计每个标签的出现次数
    label_counts = [0] * len(intent_labels)
    total_samples = len(train_df)
    
    for idx, row in train_df.iterrows():
        intent_str = str(row['intent'])
        if pd.isna(intent_str):
            continue
            
        intent_list = intent_str.split('&')
        for i, label in enumerate(intent_labels):
            if label in intent_list:
                label_counts[i] += 1
    
    print(f"Intent标签分布: {dict(zip(intent_labels, label_counts))}")
    
    # 特别处理Psychological_fulfillment
    psych_idx = intent_labels.index('Psychological_fulfillment')
    psych_count = label_counts[psych_idx]
    
    # 动态权重调整策略
    base_weights = []
    for i, count in enumerate(label_counts):
        if count > 0:
            if i == psych_idx and psych_count < 500:  # Psychological_fulfillment样本过少
                # 给予更高的权重
                weight = total_samples / (len(intent_labels) * count) * 3.0
                print(f"Psychological_fulfillment样本过少({psych_count})，应用3倍权重: {weight:.2f}")
            else:
                weight = total_samples / (len(intent_labels) * count)
        else:
            weight = 1.0
        base_weights.append(weight)
    
    print(f"增强后Intent类别权重: {dict(zip(intent_labels, [f'{w:.2f}' for w in base_weights]))}")
    
    return torch.tensor(base_weights, dtype=torch.float)

def load_and_preprocess_data():
    """加载和预处理数据，包含评论数据"""
    data_dir = "data/"
    
    # 加载CSV数据
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), sep=',', header=None,
                          names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"), sep=',', header=None,
                        names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
    test_df = pd.read_csv(os.path.join(data_dir, "test1.csv"), sep=',', header=None,
                         names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
    
    # 加载JSON数据并整合文本
    def load_json_data():
        topics_path = os.path.join(data_dir, "news_topic1.json")
        docs_path = os.path.join(data_dir, "news_docs.json")
        
        topics_data = {}
        docs_data = {}
        
        if os.path.exists(topics_path):
            with open(topics_path, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
        
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
        
        return topics_data, docs_data
    
    topics_data, docs_data = load_json_data()
    
    def get_base_text_content(row_id):
        """获取基础文本内容（不包含评论）"""
        str_id = str(row_id)
        topic_text = topics_data.get(str_id, "unknown topic")
        
        # 如果有文档数据，使用文档内容；否则只使用标题
        doc_content = ""
        if docs_data and str_id in docs_data:
            doc_content = docs_data.get(str_id, {}).get("content", "")
        
        # 如果内容为空，至少使用标题
        if not doc_content.strip():
            return topic_text
        else:
            return f"{topic_text} {doc_content}"
    
    # 为数据框添加基础文本列（后续会在EnhancedTextDataset中增强）
    train_df['text'] = train_df['id'].apply(get_base_text_content)
    val_df['text'] = val_df['id'].apply(get_base_text_content)
    test_df['text'] = test_df['id'].apply(get_base_text_content)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    return train_df, val_df, test_df

class FourteenDMINTTrainer:
    """Fourteen版训练器 - 集成评论数据和所有优化策略"""
    
    def __init__(self, model, train_loader, val_loader, device, num_epochs=10, intent_class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 损失函数 - 为多标签分类使用带权重的BCE
        self.criterion_stance = nn.CrossEntropyLoss()
        
        # 为多标签分类使用带权重的BCEWithLogitsLoss
        if intent_class_weights is not None:
            self.criterion_intent = nn.BCEWithLogitsLoss(pos_weight=intent_class_weights.to(device))
            print(f"使用加权的多标签损失函数，权重: {intent_class_weights}")
        else:
            self.criterion_intent = nn.BCEWithLogitsLoss()
            
        self.criterion_harmfulness = nn.CrossEntropyLoss()
        self.criterion_fairness = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=2e-5,
            weight_decay=0.01
        )
        
        # 使用WarmupPoly学习率调度器
        self.scheduler = WarmupPolyLR(
            self.optimizer,
            total_epochs=num_epochs,
            warmup_epochs=2,
            poly_exp=0.9,
            min_lr=1e-6,
            base_lr=2e-5
        )
        
        self.best_val_loss = float('inf')
        self.best_intent_f1 = 0.0
        self.best_psych_f1 = 0.0
        self.patience = 8
        self.counter = 0
        
        # 训练历史
        self.train_history = {
            'loss': [], 'lr': [], 'stance_acc': [], 'intent_f1': [],
            'harmfulness_acc': [], 'fairness_acc': [], 'psych_f1': []
        }
        
    def compute_balanced_loss(self, outputs, labels):
        """计算平衡的多任务损失，特别关注Psychological_fulfillment"""
        loss_stance = self.criterion_stance(outputs['stance'], labels['stance'])
        loss_intent = self.criterion_intent(outputs['intent'], labels['intent'])
        loss_harmfulness = self.criterion_harmfulness(outputs['harmfulness'], labels['harmfulness'])
        loss_fairness = self.criterion_fairness(outputs['fairness'], labels['fairness'])
        
        # 动态权重调整 - 特别关注intent和fairness
        total_loss = (1.2 * loss_stance + 2.5 * loss_intent +  # 降低intent权重
                     1.2 * loss_harmfulness + 1.8 * loss_fairness)  # 提高fairness权重
        
        return total_loss, {
            'stance': loss_stance.item(),
            'intent': loss_intent.item(),
            'harmfulness': loss_harmfulness.item(),
            'fairness': loss_fairness.item()
        }
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_batches = len(self.train_loader)
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            self.optimizer.zero_grad()
            
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            stance_labels = batch['stance'].to(self.device)
            intent_labels = batch['intent'].to(self.device)
            harmfulness_labels = batch['harmfulness'].to(self.device)
            fairness_labels = batch['fairness'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            
            # 计算平衡的多任务损失
            labels = {
                'stance': stance_labels,
                'intent': intent_labels,
                'harmfulness': harmfulness_labels,
                'fairness': fairness_labels
            }
            total_loss_batch, task_losses = self.compute_balanced_loss(outputs, labels)
            
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # 每10个batch打印一次进度
            if batch_idx % 10 == 0:
                progress = (batch_idx / total_batches) * 100
                batch_time = time.time() - batch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f'  Batch {batch_idx}/{total_batches} ({progress:.1f}%), '
                      f'Loss: {total_loss_batch.item():.4f}, '
                      f'LR: {current_lr:.2e}, '
                      f'Batch Time: {batch_time:.2f}s')
        
        # 更新学习率
        current_lr = self.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / total_batches
        return avg_loss, epoch_time, current_lr
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_batches = len(self.val_loader)
        
        all_predictions = {'stance': [], 'intent': [], 'harmfulness': [], 'fairness': []}
        all_labels = {'stance': [], 'intent': [], 'harmfulness': [], 'fairness': []}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                stance_labels = batch['stance'].to(self.device)
                intent_labels = batch['intent'].to(self.device)
                harmfulness_labels = batch['harmfulness'].to(self.device)
                fairness_labels = batch['fairness'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask)
                
                # 计算损失
                labels = {
                    'stance': stance_labels,
                    'intent': intent_labels,
                    'harmfulness': harmfulness_labels,
                    'fairness': fairness_labels
                }
                total_loss_batch, _ = self.compute_balanced_loss(outputs, labels)
                total_loss += total_loss_batch.item()
                
                # 收集预测结果
                all_predictions['stance'].extend(torch.argmax(outputs['stance'], 1).cpu().numpy())
                all_labels['stance'].extend(stance_labels.cpu().numpy())
                
                all_predictions['harmfulness'].extend(torch.argmax(outputs['harmfulness'], 1).cpu().numpy())
                all_labels['harmfulness'].extend(harmfulness_labels.cpu().numpy())
                
                all_predictions['fairness'].extend(torch.argmax(outputs['fairness'], 1).cpu().numpy())
                all_labels['fairness'].extend(fairness_labels.cpu().numpy())
                
                # Intent多标签分类 - 使用自适应阈值
                intent_probs = torch.sigmoid(outputs['intent'])
                intent_preds = (intent_probs > 0.3).float().cpu().numpy()
                all_predictions['intent'].extend(intent_preds)
                all_labels['intent'].extend(intent_labels.cpu().numpy())
        
        # 计算指标
        metrics = {}
        for task in ['stance', 'harmfulness', 'fairness']:
            if len(all_labels[task]) > 0:
                metrics[f'{task}_accuracy'] = accuracy_score(all_labels[task], all_predictions[task])
                metrics[f'{task}_f1'] = f1_score(all_labels[task], all_predictions[task], average='weighted')
            else:
                metrics[f'{task}_accuracy'] = 0.0
                metrics[f'{task}_f1'] = 0.0
        
        # Intent的多标签指标
        try:
            intent_labels_array = np.array(all_labels['intent'])
            intent_predictions_array = np.array(all_predictions['intent'])
            
            # 精确匹配准确率
            intent_accuracy = np.mean(np.all(
                intent_predictions_array == intent_labels_array, axis=1
            ))
            metrics['intent_exact_match'] = float(intent_accuracy)
            
            # 计算macro F1
            metrics['intent_macro_f1'] = f1_score(
                intent_labels_array, 
                intent_predictions_array, 
                average='macro',
                zero_division=0
            )
            
            # 计算每个intent标签的F1
            intent_f1_scores = f1_score(intent_labels_array, intent_predictions_array, average=None, zero_division=0)
            intent_labels_names = ['Political', 'Economic', 'Psychological', 'Public']
            for i, label in enumerate(intent_labels_names):
                metrics[f'intent_{label}_f1'] = float(intent_f1_scores[i])
                
        except Exception as e:
            print(f"计算intent指标时出错: {e}")
            metrics['intent_exact_match'] = 0.0
            metrics['intent_macro_f1'] = 0.0
            for label in ['Political', 'Economic', 'Psychological', 'Public']:
                metrics[f'intent_{label}_f1'] = 0.0
        
        avg_loss = total_loss / total_batches
        return avg_loss, metrics
        
    def train(self):
        """完整训练过程"""
        print("开始Fourteen版DMINT模型训练...")
        total_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            train_loss, epoch_time, current_lr = self.train_epoch(epoch + 1)
            val_loss, val_metrics = self.validate()
            
            # 更新训练历史
            self.train_history['loss'].append(train_loss)
            self.train_history['lr'].append(current_lr)
            self.train_history['stance_acc'].append(val_metrics['stance_accuracy'])
            self.train_history['intent_f1'].append(val_metrics['intent_macro_f1'])
            self.train_history['harmfulness_acc'].append(val_metrics['harmfulness_accuracy'])
            self.train_history['fairness_acc'].append(val_metrics['fairness_accuracy'])
            self.train_history['psych_f1'].append(val_metrics['intent_Psychological_f1'])
            
            # 计算总训练时间
            total_time = time.time() - total_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            
            print(f"\nEpoch {epoch+1} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  当前学习率: {current_lr:.2e}")
            print(f"  Epoch时间: {epoch_time/60:.1f}分钟")
            print(f"  总训练时间: {hours}小时{minutes}分钟")
            print(f"  验证指标:")
            for metric, value in val_metrics.items():
                print(f"    {metric}: {value:.4f}")
            
            # 早停机制 - 基于Psychological_fulfillment和整体intent F1
            current_intent_f1 = val_metrics['intent_macro_f1']
            current_psych_f1 = val_metrics['intent_Psychological_f1']
            
            improvement = False
            if current_intent_f1 > self.best_intent_f1 + 0.005:
                self.best_intent_f1 = current_intent_f1
                improvement = True
                
            if current_psych_f1 > self.best_psych_f1 + 0.01:
                self.best_psych_f1 = current_psych_f1
                improvement = True
                print(f"🎉 Psychological_fulfillment F1提升到: {current_psych_f1:.4f}")
            
            if improvement:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler,
                    'epoch': epoch,
                    'best_intent_f1': self.best_intent_f1,
                    'best_psych_f1': self.best_psych_f1,
                    'train_history': self.train_history
                }, 'outputs/fourteen_dmint_model.pth')
                print("✓ 保存最佳模型!")
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"⚠ 早停: {self.patience} 个epoch未显著提升")
                    break
            
            print("=" * 60)
        
        # 保存训练历史
        history_df = pd.DataFrame(self.train_history)
        history_df.to_csv('outputs/fourteen_training_history.csv', index=False)
        print("训练历史已保存到 outputs/fourteen_training_history.csv")

def main():
    """主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载tokenizer和模型
        print("加载模型...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        bert_model = BertModel.from_pretrained(MODEL_PATH)
        print("模型加载成功!")
        
        # 加载数据
        train_df, val_df, test_df = load_and_preprocess_data()
        
        # 计算增强的类别权重
        intent_class_weights = compute_enhanced_class_weights(train_df)
        
        # 创建数据集（包含评论数据）
        train_dataset = EnhancedTextDataset(train_df, tokenizer, phase="训练")
        val_dataset = EnhancedTextDataset(val_df, tokenizer, phase="验证")
        test_dataset = EnhancedTextDataset(test_df, tokenizer, phase="测试")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        print(f"训练集batch数量: {len(train_loader)}")
        print(f"验证集batch数量: {len(val_loader)}")
        print(f"测试集batch数量: {len(test_loader)}")
        
        # 创建Fourteen版DMINT模型
        model = FourteenDMINTModel(bert_model)
        model = model.to(device)
        print("Fourteen版DMINT模型创建成功!")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 确保输出目录存在
        os.makedirs('outputs', exist_ok=True)
        
        # 训练模型
        trainer = FourteenDMINTTrainer(
            model, train_loader, val_loader, device, 
            num_epochs=10,
            intent_class_weights=intent_class_weights
        )
        trainer.train()
        
        print("\n训练完成! 模型已保存到 outputs/fourteen_dmint_model.pth")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    
    import sys
    from datetime import datetime
    
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # 重定向标准输出到文件和控制台
    class Tee:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()  # 确保及时写入
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    # 打开日志文件
    log_f = open(log_file, 'w', encoding='utf-8')
    
    # 保存原始标准输出
    original_stdout = sys.stdout
    
    try:
        # 重定向标准输出到文件和控制台
        sys.stdout = Tee(original_stdout, log_f)
        
        print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"日志文件: {log_file}")
        print("=" * 60)
        
        # 运行主函数
        main()
        
        print("=" * 60)
        print(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复标准输出并关闭文件
        sys.stdout = original_stdout
        log_f.close()
        print(f"日志已保存到: {log_file}")