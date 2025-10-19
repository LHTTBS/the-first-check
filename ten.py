"""
增强版DMINT模型 - 主模型文件
包含模型定义、训练逻辑和数据处理
"""

import pandas as pd
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Dict, List, Tuple
import time

# 设置模型路径
MODEL_PATH = "C:/Users/LHTBS/Desktop/check919/models/chinese-roberta-wwm-ext"

class TextDataset(Dataset):
    """文本数据集类，处理多任务学习的数据加载"""
    
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
    """多尺度特征提取器 - 提取不同粒度的文本特征"""
    
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
        # 转置为卷积需要的格式 [batch_size, hidden_size, seq_len]
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        local_feat = F.relu(self.conv1(x))
        medium_feat = F.relu(self.conv2(x))
        global_feat = F.relu(self.conv3(x))
        
        # 全局平均池化
        local_pool = F.adaptive_avg_pool1d(local_feat, 1).squeeze(-1)
        medium_pool = F.adaptive_avg_pool1d(medium_feat, 1).squeeze(-1)
        global_pool = F.adaptive_avg_pool1d(global_feat, 1).squeeze(-1)
        
        # 拼接多尺度特征
        multi_scale_features = torch.cat([local_pool, medium_pool, global_pool], dim=1)
        
        # 特征融合
        fused_features = self.fusion(multi_scale_features)
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
        intent_logits_reshaped = intent_logits.unsqueeze(1)
        attended_logits, _ = self.attention(
            intent_logits_reshaped, 
            intent_logits_reshaped, 
            intent_logits_reshaped
        )
        attended_logits = attended_logits.squeeze(1)
        
        return attended_logits

class GatedIntentAggregator(nn.Module):
    """门控意图聚合器 - 动态融合不同视角的特征"""
    
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
        gate_weights = self.gate_network(concatenated)
        
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

class EnhancedDMINTModel(nn.Module):
    """增强版DMINT模型 - 专门优化多标签分类的多任务学习模型"""
    
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
            nn.Linear(feature_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # Against, Neither, Favor
        )
        
        self.harmfulness_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Harmful, Unharmful
        )
        
        self.fairness_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Tinted, Fairness
        )
        
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
    """加载和预处理数据"""
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
    
    def get_text_content(row_id):
        """获取文本内容"""
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
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=2e-5,
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
            
            # 每10个batch打印一次进度
            if batch_idx % 10 == 0:
                progress = (batch_idx / total_batches) * 100
                batch_time = time.time() - batch_start_time
                
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
        
        # 创建数据加载器
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
        
        print("\n训练完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_training()