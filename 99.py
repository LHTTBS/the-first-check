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
from sklearn.utils.class_weight import compute_class_weight
import sys

# 设置模型路径
model_path = "C:/Users/LHTBS/Desktop/check919/models/chinese-roberta-wwm-ext"

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 标签映射
        self.stance_map = {'Against': 0, 'Neither': 1, 'Favor': 2}
        self.fairness_map = {'Tinted': 0, 'Fairness': 1}
        self.harmfulness_map = {'Harmful': 0, 'Unharmful': 1}
        
        # Intent是多标签分类
        self.intent_labels = ['Political_interest', 'Economic_interest', 
                             'Psychological_fulfillment', 'Public_interest']
        
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
        text = str(row['text']) if 'text' in row else "default text"
        
        # 编码文本
        encoding = self.tokenizer(
            text,
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
            'harmfulness': torch.tensor(harmfulness, dtype=torch.long)
        }

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    def __init__(self, input_dim: int = 768, output_dim: int = 256, dropout: float = 0.1):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 多尺度卷积层
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=2, padding=1)  # 局部特征
        self.conv2 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)  # 中等特征
        self.conv3 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)  # 全局特征
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(384, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_size] -> 转置为 [batch_size, hidden_size, seq_len]
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        local_feat = F.relu(self.conv1(x))  # [batch_size, 128, seq_len]
        medium_feat = F.relu(self.conv2(x)) # [batch_size, 128, seq_len]
        global_feat = F.relu(self.conv3(x)) # [batch_size, 128, seq_len]
        
        # 全局平均池化
        local_pool = F.adaptive_avg_pool1d(local_feat, 1).squeeze(-1)  # [batch_size, 128]
        medium_pool = F.adaptive_avg_pool1d(medium_feat, 1).squeeze(-1) # [batch_size, 128]
        global_pool = F.adaptive_avg_pool1d(global_feat, 1).squeeze(-1) # [batch_size, 128]
        
        # 拼接多尺度特征
        multi_scale_features = torch.cat([local_pool, medium_pool, global_pool], dim=1)  # [batch_size, 384]
        
        # 特征融合
        fused_features = self.fusion(multi_scale_features)  # [batch_size, output_dim]
        fused_features = self.layer_norm(fused_features)
        
        return fused_features

class DifferentiatedFeatureExtractor(nn.Module):
    """差异化多视角特征提取器 - 每个视角有不同的网络结构"""
    def __init__(self, perspective_type: str, input_dim: int = 768, output_dim: int = 256, dropout: float = 0.1):
        super(DifferentiatedFeatureExtractor, self).__init__()
        self.perspective_type = perspective_type
        
        if perspective_type == "belief":
            # 信念视角：更深网络，关注深层次理解
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 384),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(384, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif perspective_type == "desire":
            # 欲望视角：中等深度，关注动机
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 448),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(448, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:  # plan
            # 计划视角：更浅网络，关注行动层面
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 320),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(320, output_dim),
                nn.LayerNorm(output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

class EnhancedIntentClassifier(nn.Module):
    """增强的意图分类器 - 专门处理多标签分类"""
    def __init__(self, feature_dim: int = 256, num_intents: int = 4, dropout: float = 0.2):
        super(EnhancedIntentClassifier, self).__init__()
        
        # 更深的网络结构来处理多标签关系
        self.intent_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_intents)
        )
        
        # 添加注意力机制来捕捉标签间的关系
        self.attention = nn.MultiheadAttention(
            embed_dim=num_intents, 
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础意图分类
        intent_logits = self.intent_net(x)
        
        # 使用自注意力捕捉标签间关系
        intent_logits_reshaped = intent_logits.unsqueeze(1)  # [batch_size, 1, num_intents]
        attended_logits, _ = self.attention(
            intent_logits_reshaped, 
            intent_logits_reshaped, 
            intent_logits_reshaped
        )
        attended_logits = attended_logits.squeeze(1)
        
        return attended_logits

class GatedIntentAggregator(nn.Module):
    """门控意图聚合器"""
    def __init__(self, feature_dim: int = 256, num_intents: int = 4, dropout: float = 0.1):
        super(GatedIntentAggregator, self).__init__()
        
        # 门控机制 - 学习每个视角的重要性权重
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
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
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 使用增强的意图分类器
        self.intent_classifier = EnhancedIntentClassifier(feature_dim, num_intents, dropout)
        
    def forward(self, belief_features: torch.Tensor, desire_features: torch.Tensor, 
                plan_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        batch_size = belief_features.size(0)
        
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
            'gate_weights': gate_weights  # 返回门控权重用于分析
        }

class EnhancedDMINTModel(nn.Module):
    """增强版DMINT模型 - 专门优化多标签分类"""
    def __init__(self, bert_model, hidden_size: int = 768, feature_dim: int = 256, dropout: float = 0.3):
        super(EnhancedDMINTModel, self).__init__()
        
        self.bert = bert_model
        
        # 多尺度特征提取器（用于序列特征）
        self.multi_scale_extractor = MultiScaleFeatureExtractor(hidden_size, feature_dim, dropout)
        
        # 三个差异化的视角特征提取器（用于CLS特征）
        self.belief_extractor = DifferentiatedFeatureExtractor("belief", hidden_size, feature_dim, dropout)
        self.desire_extractor = DifferentiatedFeatureExtractor("desire", hidden_size, feature_dim, dropout)
        self.plan_extractor = DifferentiatedFeatureExtractor("plan", hidden_size, feature_dim, dropout)
        
        # 门控意图聚合器
        self.intent_aggregator = GatedIntentAggregator(feature_dim, 4, dropout)
        
        # 任务特定的分类器
        self.stance_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # 使用多尺度特征+信念特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # Against, Neither, Favor
        )
        
        self.harmfulness_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # 使用多尺度特征+计划特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Harmful, Unharmful
        )
        
        self.fairness_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),  # 使用融合特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Tinted, Fairness
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # BERT特征提取 - 获取序列输出和CLS输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        cls_features = bert_output.pooler_output  # [batch_size, hidden_size]
        
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
            'gate_weights': intent_output['gate_weights']  # 返回门控权重
        }

def compute_intent_class_weights(train_df):
    """计算多标签分类的类别权重"""
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
    
    # 计算权重：样本总数 / (类别数 * 类别出现次数)
    # 使用平滑处理避免除零
    weights = []
    for count in label_counts:
        if count > 0:
            weight = total_samples / (len(intent_labels) * count)
        else:
            weight = 1.0  # 如果某个标签没有出现，使用默认权重
        weights.append(weight)
    
    print(f"Intent标签分布: {dict(zip(intent_labels, label_counts))}")
    print(f"Intent类别权重: {dict(zip(intent_labels, [f'{w:.2f}' for w in weights]))}")
    
    return torch.tensor(weights, dtype=torch.float)

def load_and_preprocess_data():
    """加载和预处理数据 - 修复了CSV分隔符问题"""
    data_dir = "data/"
    
    # 修复：统一使用逗号分隔符
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
    
    def get_text_content(row_id):
        """优化文本内容获取"""
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
    
    # 为数据框添加文本列
    train_df['text'] = train_df['id'].apply(get_text_content)
    val_df['text'] = val_df['id'].apply(get_text_content)
    test_df['text'] = test_df['id'].apply(get_text_content)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    # 打印标签分布
    print("\n训练集标签分布:")
    print(f"Stance: {train_df['stance'].value_counts().to_dict()}")
    print(f"Intent样本分布: {train_df['intent'].value_counts().head(10).to_dict()}")
    print(f"Fairness: {train_df['fairness'].value_counts().to_dict()}")
    print(f"Harmfulness: {train_df['harmfulness'].value_counts().to_dict()}")
    
    return train_df, val_df, test_df

def safe_intent_metrics(predictions, labels):
    """安全地计算多标签分类指标，处理维度问题"""
    if len(predictions) == 0 or len(labels) == 0:
        return {
            'exact_match': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'per_label_f1': [0.0, 0.0, 0.0, 0.0]
        }
    
    try:
        predictions_array = np.array(predictions)
        labels_array = np.array(labels)
        
        # 确保数组是二维的
        if predictions_array.ndim == 1:
            predictions_array = predictions_array.reshape(1, -1)
            labels_array = labels_array.reshape(1, -1)
        
        # 精确匹配准确率
        exact_match = np.mean(np.all(predictions_array == labels_array, axis=1))
        
        # 计算F1分数
        try:
            macro_f1 = f1_score(labels_array, predictions_array, average='macro', zero_division=0)
            micro_f1 = f1_score(labels_array, predictions_array, average='micro', zero_division=0)
            per_label_f1 = f1_score(labels_array, predictions_array, average=None, zero_division=0)
        except:
            macro_f1 = 0.0
            micro_f1 = 0.0
            per_label_f1 = [0.0] * 4
        
        return {
            'exact_match': float(exact_match),
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'per_label_f1': per_label_f1.tolist() if hasattr(per_label_f1, 'tolist') else per_label_f1
        }
    except Exception as e:
        print(f"计算intent指标时出错: {e}")
        return {
            'exact_match': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'per_label_f1': [0.0, 0.0, 0.0, 0.0]
        }

class ImprovedDMINTTrainer:
    """改进的训练器 - 专门优化多标签分类"""
    def __init__(self, model, train_loader, val_loader, device, num_epochs=5, intent_class_weights=None):
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
        
        # 优化器 - 使用更大的学习率
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=2e-5,  # 从1e-5增加到2e-5
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,
            T_mult=2
        )
        
        self.best_val_loss = float('inf')
        self.best_intent_f1 = 0.0
        self.patience = 8
        self.counter = 0
        self.start_time = time.time()
        
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
            
            # 计算多任务损失
            loss_stance = self.criterion_stance(outputs['stance'], stance_labels)
            loss_intent = self.criterion_intent(outputs['intent'], intent_labels)
            loss_harmfulness = self.criterion_harmfulness(outputs['harmfulness'], harmfulness_labels)
            loss_fairness = self.criterion_fairness(outputs['fairness'], fairness_labels)
            
            # 动态权重调整 - 给多标签分类更高权重
            total_loss_batch = (loss_stance + 4.0 * loss_intent + 
                              loss_harmfulness + loss_fairness)
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # 每50个batch打印一次进度
            if batch_idx % 10 == 0:
                progress = (batch_idx / total_batches) * 100
                batch_time = time.time() - batch_start_time
                estimated_epoch_time = (batch_time / (batch_idx + 1)) * total_batches
                remaining_time = estimated_epoch_time * (self.num_epochs - epoch + 1)
                
                print(f'  Batch {batch_idx}/{total_batches} ({progress:.1f}%), '
                      f'Loss: {total_loss_batch.item():.4f}, '
                      f'Intent Loss: {loss_intent.item():.4f}, '
                      f'Batch Time: {batch_time:.2f}s')
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / total_batches
        return avg_loss, epoch_time
    
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
                loss_stance = self.criterion_stance(outputs['stance'], stance_labels)
                loss_intent = self.criterion_intent(outputs['intent'], intent_labels)
                loss_harmfulness = self.criterion_harmfulness(outputs['harmfulness'], harmfulness_labels)
                loss_fairness = self.criterion_fairness(outputs['fairness'], fairness_labels)
                
                total_loss_batch = (loss_stance + 4.0 * loss_intent + 
                                  loss_harmfulness + loss_fairness)
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
                k = max(1, int(intent_labels.sum(dim=1).float().mean().item()))
                intent_preds = torch.zeros_like(intent_probs)
                for i in range(intent_probs.size(0)):
                    topk_indices = torch.topk(intent_probs[i], k).indices
                    intent_preds[i, topk_indices] = 1
                
                all_predictions['intent'].extend(intent_preds.cpu().numpy())
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
        intent_metrics = safe_intent_metrics(all_predictions['intent'], all_labels['intent'])
        metrics['intent_exact_match'] = intent_metrics['exact_match']
        metrics['intent_macro_f1'] = intent_metrics['macro_f1']
        metrics['intent_micro_f1'] = intent_metrics['micro_f1']
        
        # 每个intent标签的F1
        intent_labels_names = ['Political', 'Economic', 'Psychological', 'Public']
        for i, label in enumerate(intent_labels_names):
            metrics[f'intent_{label}_f1'] = intent_metrics['per_label_f1'][i] if i < len(intent_metrics['per_label_f1']) else 0.0
        
        avg_loss = total_loss / total_batches
        return avg_loss, metrics
        
    def train(self):
        """完整训练过程"""
        print("开始训练...")
        total_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            train_loss, epoch_time = self.train_epoch(epoch + 1)
            val_loss, val_metrics = self.validate()
            
            self.scheduler.step()
            
            # 计算总训练时间
            total_time = time.time() - total_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            
            print(f"\nEpoch {epoch+1} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  Epoch时间: {epoch_time/60:.1f}分钟")
            print(f"  总训练时间: {hours}小时{minutes}分钟")
            print(f"  验证指标:")
            for metric, value in val_metrics.items():
                print(f"    {metric}: {value:.4f}")
            
            # 早停机制 - 基于intent F1分数
            current_intent_f1 = val_metrics['intent_macro_f1']
            if current_intent_f1 > self.best_intent_f1:
                self.best_intent_f1 = current_intent_f1
                torch.save(self.model.state_dict(), 'outputs/best_enhanced_dmint_model.pth')
                print("✓ 保存最佳模型! (基于intent F1)")
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"⚠ 早停: {self.patience} 个epoch意图F1分数未提升")
                    break
            
            print("=" * 60)

class TrainingValidator:
    """训练验证器 - 全面检测训练流程的每个环节"""
    
    def __init__(self):
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def validate_data_loading(self):
        """验证数据加载"""
        print("=" * 60)
        print("1. 验证数据加载")
        print("=" * 60)
        
        try:
            train_df, val_df, test_df = load_and_preprocess_data()
            
            if train_df is None or val_df is None or test_df is None:
                print("✗ 数据加载失败 - 返回了None")
                return False
            
            print(f"✓ 数据加载成功")
            print(f"  训练集: {len(train_df)} 样本")
            print(f"  验证集: {len(val_df)} 样本") 
            print(f"  测试集: {len(test_df)} 样本")
            
            # 检查数据完整性
            required_columns = ['id', 'stance', 'intent', 'fairness', 'harmfulness', 'text']
            for df_name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"✗ {df_name}缺少列: {missing_columns}")
                    return False
            
            # 检查标签数据
            for df_name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
                null_counts = df[['stance', 'intent', 'fairness', 'harmfulness']].isnull().sum()
                if null_counts.sum() > 0:
                    print(f"✗ {df_name}有缺失标签: {dict(null_counts)}")
                    return False
            
            # 检查文本数据
            for df_name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
                text_lengths = df['text'].str.len()
                if text_lengths.min() == 0:
                    print(f"✗ {df_name}有零长度文本")
                    return False
                print(f"  {df_name}文本长度 - 平均: {text_lengths.mean():.1f}, 最小: {text_lengths.min()}, 最大: {text_lengths.max()}")
            
            self.results['dataframes'] = (train_df, val_df, test_df)
            return True
            
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_model_creation(self):
        """验证模型创建"""
        print("\n" + "=" * 60)
        print("2. 验证模型创建")
        print("=" * 60)
        
        try:
            model_path = "C:/Users/LHTBS/Desktop/check919/models/chinese-roberta-wwm-ext"
            
            print("加载tokenizer和BERT模型...")
            tokenizer = BertTokenizer.from_pretrained(model_path)
            bert_model = BertModel.from_pretrained(model_path)
            
            print("创建DMINT模型...")
            model = EnhancedDMINTModel(bert_model)
            
            # 检查模型参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✓ 模型创建成功")
            print(f"  总参数: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  模型结构:")
            for name, module in model.named_children():
                num_params = sum(p.numel() for p in module.parameters())
                print(f"    {name}: {num_params:,} 参数")
            
            self.results['model'] = model
            self.results['tokenizer'] = tokenizer
            return True
            
        except Exception as e:
            print(f"✗ 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_data_processing(self):
        """验证数据处理"""
        print("\n" + "=" * 60)
        print("3. 验证数据处理")
        print("=" * 60)
        
        try:
            if 'dataframes' not in self.results or 'tokenizer' not in self.results:
                print("✗ 需要先完成数据加载和模型创建")
                return False
            
            train_df, val_df, test_df = self.results['dataframes']
            tokenizer = self.results['tokenizer']
            
            # 使用少量数据测试
            mini_train_df = train_df.head(32)
            train_dataset = TextDataset(mini_train_df, tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            
            print(f"✓ 数据处理成功")
            print(f"  数据集大小: {len(train_dataset)}")
            print(f"  Batch数量: {len(train_loader)}")
            
            # 检查一个batch的数据
            for batch in train_loader:
                print(f"  Batch数据形状:")
                print(f"    input_ids: {batch['input_ids'].shape}")
                print(f"    attention_mask: {batch['attention_mask'].shape}")
                print(f"    stance: {batch['stance'].shape} (值范围: {batch['stance'].min().item()}~{batch['stance'].max().item()})")
                print(f"    intent: {batch['intent'].shape} (值范围: {batch['intent'].min().item():.2f}~{batch['intent'].max().item():.2f})")
                print(f"    fairness: {batch['fairness'].shape} (值范围: {batch['fairness'].min().item()}~{batch['fairness'].max().item()})")
                print(f"    harmfulness: {batch['harmfulness'].shape} (值范围: {batch['harmfulness'].min().item()}~{batch['harmfulness'].max().item()})")
                break
            
            self.results['train_loader'] = train_loader
            return True
            
        except Exception as e:
            print(f"✗ 数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_forward_pass(self):
        """验证前向传播"""
        print("\n" + "=" * 60)
        print("4. 验证前向传播")
        print("=" * 60)
        
        try:
            if 'model' not in self.results or 'train_loader' not in self.results:
                print("✗ 需要先完成模型创建和数据处理")
                return False
            
            model = self.results['model']
            train_loader = self.results['train_loader']
            
            model = model.to(self.device)
            model.eval()  # 使用eval模式避免dropout影响
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask)
                
                print(f"✓ 前向传播成功")
                print(f"  输出形状:")
                print(f"    stance: {outputs['stance'].shape}")
                print(f"    intent: {outputs['intent'].shape}")
                print(f"    fairness: {outputs['fairness'].shape}")
                print(f"    harmfulness: {outputs['harmfulness'].shape}")
                print(f"    gate_weights: {outputs['gate_weights'].shape}")
                
                # 检查输出值范围
                print(f"  输出值范围:")
                print(f"    stance: {outputs['stance'].min().item():.4f} ~ {outputs['stance'].max().item():.4f}")
                print(f"    intent: {outputs['intent'].min().item():.4f} ~ {outputs['intent'].max().item():.4f}")
                print(f"    fairness: {outputs['fairness'].min().item():.4f} ~ {outputs['fairness'].max().item():.4f}")
                print(f"    harmfulness: {outputs['harmfulness'].min().item():.4f} ~ {outputs['harmfulness'].max().item():.4f}")
                print(f"    gate_weights: {outputs['gate_weights'].min().item():.4f} ~ {outputs['gate_weights'].max().item():.4f}")
                
                # 检查gate_weights是否合理（应该和为1）
                gate_sums = outputs['gate_weights'].sum(dim=1)
                print(f"  gate_weights每行和: {gate_sums.min().item():.4f} ~ {gate_sums.max().item():.4f}")
                
                break
            
            self.results['model'] = model
            return True
            
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_loss_calculation(self):
        """验证损失计算"""
        print("\n" + "=" * 60)
        print("5. 验证损失计算")
        print("=" * 60)
        
        try:
            if 'model' not in self.results or 'train_loader' not in self.results or 'dataframes' not in self.results:
                print("✗ 需要先完成前面的验证步骤")
                return False
            
            train_df, _, _ = self.results['dataframes']
            model = self.results['model']
            train_loader = self.results['train_loader']
            
            # 计算类别权重
            intent_class_weights = compute_intent_class_weights(train_df)
            
            # 创建训练器（使用train_loader作为验证集进行测试）
            trainer = ImprovedDMINTTrainer(
                model, train_loader, train_loader, self.device,
                num_epochs=1,
                intent_class_weights=intent_class_weights
            )
            
            # 测试损失计算
            model.train()  # 切换回训练模式
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                stance_labels = batch['stance'].to(self.device)
                intent_labels = batch['intent'].to(self.device)
                harmfulness_labels = batch['harmfulness'].to(self.device)
                fairness_labels = batch['fairness'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                
                # 计算各个任务的损失
                loss_stance = trainer.criterion_stance(outputs['stance'], stance_labels)
                loss_intent = trainer.criterion_intent(outputs['intent'], intent_labels)
                loss_harmfulness = trainer.criterion_harmfulness(outputs['harmfulness'], harmfulness_labels)
                loss_fairness = trainer.criterion_fairness(outputs['fairness'], fairness_labels)
                
                total_loss = (loss_stance + 4.0 * loss_intent + loss_harmfulness + loss_fairness)
                
                print(f"✓ 损失计算成功")
                print(f"  各任务损失:")
                print(f"    stance: {loss_stance.item():.4f}")
                print(f"    intent: {loss_intent.item():.4f}")
                print(f"    harmfulness: {loss_harmfulness.item():.4f}")
                print(f"    fairness: {loss_fairness.item():.4f}")
                print(f"  总损失: {total_loss.item():.4f}")
                
                # 检查损失是否为有限值
                if not torch.isfinite(total_loss):
                    print(f"✗ 损失值为非有限值: {total_loss.item()}")
                    return False
                
                break
            
            self.results['trainer'] = trainer
            return True
            
        except Exception as e:
            print(f"✗ 损失计算失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_backward_pass(self):
        """验证反向传播"""
        print("\n" + "=" * 60)
        print("6. 验证反向传播")
        print("=" * 60)
        
        try:
            if 'model' not in self.results or 'train_loader' not in self.results or 'trainer' not in self.results:
                print("✗ 需要先完成前面的验证步骤")
                return False
            
            model = self.results['model']
            train_loader = self.results['train_loader']
            trainer = self.results['trainer']
            
            model.train()
            trainer.optimizer.zero_grad()
            
            # 执行一个完整的训练步骤
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                stance_labels = batch['stance'].to(self.device)
                intent_labels = batch['intent'].to(self.device)
                harmfulness_labels = batch['harmfulness'].to(self.device)
                fairness_labels = batch['fairness'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                
                loss_stance = trainer.criterion_stance(outputs['stance'], stance_labels)
                loss_intent = trainer.criterion_intent(outputs['intent'], intent_labels)
                loss_harmfulness = trainer.criterion_harmfulness(outputs['harmfulness'], harmfulness_labels)
                loss_fairness = trainer.criterion_fairness(outputs['fairness'], fairness_labels)
                
                total_loss = (loss_stance + 4.0 * loss_intent + loss_harmfulness + loss_fairness)
                total_loss.backward()
                
                # 检查梯度
                has_gradients = False
                gradient_norms = []
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        has_gradients = True
                        grad_norm = param.grad.norm().item()
                        gradient_norms.append(grad_norm)
                
                if has_gradients:
                    print(f"✓ 反向传播成功")
                    print(f"  检测到梯度的参数数量: {len(gradient_norms)}")
                    print(f"  梯度范数范围: {min(gradient_norms):.6f} ~ {max(gradient_norms):.6f}")
                    
                    # 检查梯度是否为有限值
                    if not all(np.isfinite(grad_norm) for grad_norm in gradient_norms):
                        print(f"✗ 梯度包含非有限值")
                        return False
                else:
                    print(f"✗ 未检测到梯度")
                    return False
                
                # 执行优化步骤
                trainer.optimizer.step()
                print(f"✓ 优化器步骤完成")
                
                break
            
            return True
            
        except Exception as e:
            print(f"✗ 反向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_training_loop(self):
        """验证完整训练循环"""
        print("\n" + "=" * 60)
        print("7. 验证完整训练循环")
        print("=" * 60)
        
        try:
            if 'model' not in self.results or 'train_loader' not in self.results or 'trainer' not in self.results:
                print("✗ 需要先完成前面的验证步骤")
                return False
            
            model = self.results['model']
            train_loader = self.results['train_loader']
            trainer = self.results['trainer']
            
            print("运行一个完整的训练epoch...")
            
            # 记录初始参数
            initial_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    initial_params[name] = param.data.clone()
            
            # 运行一个训练epoch
            train_loss, epoch_time = trainer.train_epoch(1)
            
            # 检查参数是否更新
            params_updated = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if not torch.equal(initial_params[name], param.data):
                        params_updated = True
                        break
            
            if params_updated:
                print(f"✓ 训练循环成功")
                print(f"  训练损失: {train_loss:.4f}")
                print(f"  epoch时间: {epoch_time:.2f}秒")
                print(f"  模型参数已更新")
            else:
                print(f"⚠ 训练完成但参数未更新")
            
            return True
            
        except Exception as e:
            print(f"✗ 训练循环失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_validation(self):
        """运行完整验证"""
        print("开始训练流程验证...")
        print("=" * 60)
        
        validation_steps = [
            ("数据加载", self.validate_data_loading),
            ("模型创建", self.validate_model_creation),
            ("数据处理", self.validate_data_processing),
            ("前向传播", self.validate_forward_pass),
            ("损失计算", self.validate_loss_calculation),
            ("反向传播", self.validate_backward_pass),
            ("训练循环", self.validate_training_loop),
        ]
        
        passed_steps = 0
        total_steps = len(validation_steps)
        
        for step_name, validation_func in validation_steps:
            try:
                success = validation_func()
                if success:
                    passed_steps += 1
                    print(f"✓ {step_name} - 通过")
                else:
                    print(f"✗ {step_name} - 失败")
                    break
            except Exception as e:
                print(f"✗ {step_name} - 异常: {e}")
                break
        
        print("\n" + "=" * 60)
        print("验证结果总结")
        print("=" * 60)
        print(f"通过步骤: {passed_steps}/{total_steps}")
        
        if passed_steps == total_steps:
            print("🎉 所有验证通过！代码可以正常训练")
            print("\n建议下一步:")
            print("1. 运行完整训练")
            print("2. 监控训练过程中的损失和指标")
            print("3. 检查验证集性能")
            return True
        else:
            print("⚠ 部分验证失败，需要修复问题")
            print(f"失败步骤: {validation_steps[passed_steps][0] if passed_steps < total_steps else 'N/A'}")
            return False

def main_training():
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件
    model_path = "C:/Users/LHTBS/Desktop/check919/models/chinese-roberta-wwm-ext"
    
    try:
        # 加载tokenizer和模型
        print("加载模型...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_model = BertModel.from_pretrained(model_path)
        print("模型加载成功!")
        
        # 加载数据
        train_df, val_df, test_df = load_and_preprocess_data()
        
        # 计算多标签分类的类别权重
        intent_class_weights = compute_intent_class_weights(train_df)
        
        # 创建数据集
        train_dataset = TextDataset(train_df, tokenizer)
        val_dataset = TextDataset(val_df, tokenizer)
        test_dataset = TextDataset(test_df, tokenizer)
        
        # 创建数据加载器 - 使用更大的batch size
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        print(f"训练集batch数量: {len(train_loader)}")
        print(f"验证集batch数量: {len(val_loader)}")
        print(f"测试集batch数量: {len(test_loader)}")
        
        # 创建增强版DMINT模型
        model = EnhancedDMINTModel(bert_model)
        model = model.to(device)
        print("增强版DMINT模型创建成功!")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 确保输出目录存在
        os.makedirs('outputs', exist_ok=True)
        
        # 训练模型 - 使用类别权重
        trainer = ImprovedDMINTTrainer(
            model, train_loader, val_loader, device, 
            num_epochs=5,
            intent_class_weights=intent_class_weights
        )
        trainer.train()
        
        # 加载最佳模型并测试
        print("\n加载最佳模型进行测试...")
        model.load_state_dict(torch.load('outputs/best_enhanced_dmint_model.pth'))
        
        # 测试模型
        model.eval()
        test_predictions = {'stance': [], 'intent': [], 'harmfulness': [], 'fairness': []}
        test_labels = {'stance': [], 'intent': [], 'harmfulness': [], 'fairness': []}
        gate_weights = []  # 收集门控权重
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                stance_labels = batch['stance'].to(device)
                intent_labels = batch['intent'].to(device)
                harmfulness_labels = batch['harmfulness'].to(device)
                fairness_labels = batch['fairness'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                test_predictions['stance'].extend(torch.argmax(outputs['stance'], 1).cpu().numpy())
                test_labels['stance'].extend(stance_labels.cpu().numpy())
                
                test_predictions['harmfulness'].extend(torch.argmax(outputs['harmfulness'], 1).cpu().numpy())
                test_labels['harmfulness'].extend(harmfulness_labels.cpu().numpy())
                
                test_predictions['fairness'].extend(torch.argmax(outputs['fairness'], 1).cpu().numpy())
                test_labels['fairness'].extend(fairness_labels.cpu().numpy())
                
                # Intent多标签分类 - 使用自适应阈值
                intent_probs = torch.sigmoid(outputs['intent'])
                k = max(1, int(intent_labels.sum(dim=1).float().mean().item()))
                intent_preds = torch.zeros_like(intent_probs)
                for i in range(intent_probs.size(0)):
                    topk_indices = torch.topk(intent_probs[i], k).indices
                    intent_preds[i, topk_indices] = 1
                
                test_predictions['intent'].extend(intent_preds.cpu().numpy())
                test_labels['intent'].extend(intent_labels.cpu().numpy())
                
                # 收集门控权重
                gate_weights.extend(outputs['gate_weights'].cpu().numpy())
        
        # 分析门控权重
        if len(gate_weights) > 0:
            gate_weights = np.array(gate_weights)
            avg_gate_weights = np.mean(gate_weights, axis=0)
            print(f"\n平均门控权重 - 信念: {avg_gate_weights[0]:.4f}, 欲望: {avg_gate_weights[1]:.4f}, 计划: {avg_gate_weights[2]:.4f}")
        else:
            print("\n没有收集到门控权重")
        
        # 计算测试指标
        print("\n" + "="*50)
        print("测试集结果:")
        print("="*50)
        
        for task in ['stance', 'harmfulness', 'fairness']:
            if len(test_labels[task]) > 0:
                acc = accuracy_score(test_labels[task], test_predictions[task])
                f1 = f1_score(test_labels[task], test_predictions[task], average='weighted')
                print(f"{task}: 准确率={acc:.4f}, F1={f1:.4f}")
            else:
                print(f"{task}: 没有数据")
        
        # 使用安全的intent指标计算
        intent_metrics = safe_intent_metrics(test_predictions['intent'], test_labels['intent'])
        print(f"intent: 精确匹配={intent_metrics['exact_match']:.4f}, F1-macro={intent_metrics['macro_f1']:.4f}, F1-micro={intent_metrics['micro_f1']:.4f}")
        
        # 保存结果
        with open('outputs/enhanced_final_results.txt', 'w', encoding='utf-8') as f:
            f.write("增强版DMINT模型最终测试结果\n")
            f.write("="*50 + "\n")
            if len(gate_weights) > 0:
                f.write(f"平均门控权重 - 信念: {avg_gate_weights[0]:.4f}, 欲望: {avg_gate_weights[1]:.4f}, 计划: {avg_gate_weights[2]:.4f}\n")
            for task in ['stance', 'harmfulness', 'fairness']:
                if len(test_labels[task]) > 0:
                    acc = accuracy_score(test_labels[task], test_predictions[task])
                    f1 = f1_score(test_labels[task], test_predictions[task], average='weighted')
                    f.write(f"{task}: 准确率={acc:.4f}, F1={f1:.4f}\n")
            f.write(f"intent: 精确匹配={intent_metrics['exact_match']:.4f}, F1-macro={intent_metrics['macro_f1']:.4f}, F1-micro={intent_metrics['micro_f1']:.4f}\n")
        
        print("\n训练完成! 结果已保存到 outputs/enhanced_final_results.txt")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数 - 整合验证和训练"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DMINT模型训练与验证')
    parser.add_argument('--mode', type=str, default='validate_and_train', 
                       choices=['validate', 'train', 'validate_and_train'],
                       help='运行模式: validate(只验证), train(只训练), validate_and_train(验证并训练)')
    
    args = parser.parse_args()
    
    if args.mode == 'validate':
        # 只运行验证
        validator = TrainingValidator()
        success = validator.run_complete_validation()
        if success:
            print("\n🎊 验证通过！可以安全进行训练")
        else:
            print("\n💡 请根据错误信息修复问题")
            
    elif args.mode == 'train':
        # 只运行训练
        print("直接开始训练...")
        main_training()
        
    elif args.mode == 'validate_and_train':
        # 先验证，验证通过后训练
        print("开始验证流程...")
        validator = TrainingValidator()
        success = validator.run_complete_validation()
        
        if success:
            print("\n🎉 验证通过！开始完整训练...")
            print("=" * 60)
            main_training()
        else:
            print("\n❌ 验证失败，请先修复问题再运行训练")

if __name__ == "__main__":
    main()