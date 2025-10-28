# intent_classifier_complete.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
from sklearn.utils.class_weight import compute_class_weight

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器 - 使用CNN提取不同粒度的文本特征"""
    def __init__(self, input_dim: int = 768, output_dim: int = 256, dropout: float = 0.1):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 多尺度卷积层：捕捉不同范围的上下文信息
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=2, padding=1)   # 局部特征（2-gram）
        self.conv2 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)   # 中等特征（3-gram）
        self.conv3 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)   # 全局特征（5-gram）
        
        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Linear(384, 512),
            nn.GELU(),  # 使用GELU激活函数，效果优于ReLU
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状转换: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, seq_len]
        x = x.transpose(1, 2)
        
        # 多尺度卷积特征提取
        local_feat = F.relu(self.conv1(x))   # 局部模式
        medium_feat = F.relu(self.conv2(x))  # 中等模式
        global_feat = F.relu(self.conv3(x))  # 全局模式
        
        # 全局平均池化 - 将序列维度压缩为1
        local_pool = F.adaptive_avg_pool1d(local_feat, 1).squeeze(-1)
        medium_pool = F.adaptive_avg_pool1d(medium_feat, 1).squeeze(-1)
        global_pool = F.adaptive_avg_pool1d(global_feat, 1).squeeze(-1)
        
        # 拼接多尺度特征
        multi_scale_features = torch.cat([local_pool, medium_pool, global_pool], dim=1)
        
        # 特征融合和降维
        fused_features = self.fusion(multi_scale_features)
        fused_features = self.layer_norm(fused_features)
        
        return fused_features

class DifferentiatedFeatureExtractor(nn.Module):
    """差异化多视角特征提取器 - 模拟人类认知的不同视角"""
    def __init__(self, perspective_type: str, input_dim: int = 768, output_dim: int = 256, dropout: float = 0.1):
        super(DifferentiatedFeatureExtractor, self).__init__()
        self.perspective_type = perspective_type
        
        # 根据不同视角设计不同的网络结构
        if perspective_type == "belief":
            # 信念视角：更深网络，关注深层次理解和价值观
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
            # 欲望视角：中等深度，关注动机和情感倾向
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 448),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(448, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:  # plan
            # 计划视角：更浅网络，关注行动层面和具体计划
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 320),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(320, output_dim),
                nn.LayerNorm(output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

class LabelAwareAttention(nn.Module):
    """标签感知注意力机制 - 捕捉标签间的依赖关系"""
    def __init__(self, num_labels: int, hidden_dim: int = 128, dropout: float = 0.2):
        super(LabelAwareAttention, self).__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        
        # 标签特定的查询向量
        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_dim))
        
        # 注意力网络
        self.attention_net = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化参数
        nn.init.xavier_uniform_(self.label_queries)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch_size, feature_dim]
        batch_size = features.size(0)
        
        # 扩展标签查询到batch维度
        label_queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 将特征转换为键值对
        features = features.unsqueeze(1).expand(-1, self.num_labels, -1)
        
        # 应用注意力机制
        attended_features, attention_weights = self.attention_net(
            label_queries, features, features
        )
        
        # 残差连接和层归一化
        attended_features = self.layer_norm(attended_features + label_queries)
        
        return attended_features

class EnhancedIntentClassifier(nn.Module):
    """增强的意图分类器 - 专门优化多标签分类"""
    def __init__(self, feature_dim: int = 256, num_intents: int = 4, dropout: float = 0.2):
        super(EnhancedIntentClassifier, self).__init__()
        self.num_intents = num_intents
        
        # 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 标签感知注意力
        self.label_attention = LabelAwareAttention(num_intents, 256, dropout)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_intents)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征增强
        enhanced_features = self.feature_enhancer(x)
        
        # 标签感知注意力
        attended_features = self.label_attention(enhanced_features)
        
        # 分类预测
        logits = self.classifier(attended_features)
        
        return logits

class GatedIntentAggregator(nn.Module):
    """门控意图聚合器 - 动态融合多视角特征"""
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
            'gate_weights': gate_weights
        }

class AdvancedIntentModel(nn.Module):
    """高级意图分类模型 - 整合所有优化组件"""
    def __init__(self, bert_model, hidden_size: int = 768, feature_dim: int = 256, 
                 num_intents: int = 4, dropout: float = 0.3):
        super(AdvancedIntentModel, self).__init__()
        
        self.bert = bert_model
        
        # 多尺度特征提取器
        self.multi_scale_extractor = MultiScaleFeatureExtractor(hidden_size, feature_dim, dropout)
        
        # 三个差异化的视角特征提取器
        self.belief_extractor = DifferentiatedFeatureExtractor("belief", hidden_size, feature_dim, dropout)
        self.desire_extractor = DifferentiatedFeatureExtractor("desire", hidden_size, feature_dim, dropout)
        self.plan_extractor = DifferentiatedFeatureExtractor("plan", hidden_size, feature_dim, dropout)
        
        # 门控意图聚合器
        self.intent_aggregator = GatedIntentAggregator(feature_dim, num_intents, dropout)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_intents)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # BERT特征提取
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        cls_features = bert_output.pooler_output
        
        # 多尺度特征提取
        multi_scale_features = self.multi_scale_extractor(sequence_output)
        
        # 三个视角的特征提取
        belief_features = self.belief_extractor(cls_features)
        desire_features = self.desire_extractor(cls_features)
        plan_features = self.plan_extractor(cls_features)
        
        # 应用dropout
        belief_features = self.dropout(belief_features)
        desire_features = self.dropout(desire_features)
        plan_features = self.dropout(plan_features)
        
        # 门控意图聚合
        intent_output = self.intent_aggregator(belief_features, desire_features, plan_features)
        
        # 最终输出
        final_logits = self.output_projection(intent_output['fused_features'])
        
        return {
            'intent_logits': final_logits,
            'gate_weights': intent_output['gate_weights'],
            'belief_features': belief_features,
            'desire_features': desire_features,
            'plan_features': plan_features,
            'multi_scale_features': multi_scale_features
        }

class IntentDataset(Dataset):
    """意图分类数据集"""
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 意图标签定义
        self.intent_labels = ['Political_interest', 'Economic_interest', 
                             'Psychological_fulfillment', 'Public_interest']
        
    def __len__(self):
        return len(self.data)
    
    def encode_intent(self, intent_str):
        """将多标签intent编码为二进制向量"""
        if pd.isna(intent_str) or intent_str == '':
            return [0] * len(self.intent_labels)
        
        intent_list = str(intent_str).split('&')
        encoding = [0] * len(self.intent_labels)
        
        for i, label in enumerate(self.intent_labels):
            if label in intent_list:
                encoding[i] = 1
                
        return encoding
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text']) if 'text' in row else str(row.get('content', 'default text'))
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 获取意图标签
        intent = self.encode_intent(row['intent'])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'intent': torch.tensor(intent, dtype=torch.float)
        }

class IntentDataProcessor:
    """意图数据处理器"""
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.intent_labels = ['Political_interest', 'Economic_interest', 
                             'Psychological_fulfillment', 'Public_interest']
    
    def load_data(self):
        """加载和预处理数据"""
        try:
            # 加载CSV数据
            train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"), sep=',', header=None,
                                  names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
            val_df = pd.read_csv(os.path.join(self.data_dir, "val.csv"), sep=',', header=None,
                                names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
            test_df = pd.read_csv(os.path.join(self.data_dir, "test1.csv"), sep=',', header=None,
                                 names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
            
            # 加载文本内容
            train_df = self._add_text_content(train_df)
            val_df = self._add_text_content(val_df)
            test_df = self._add_text_content(test_df)
            
            print(f"数据加载成功: 训练集{len(train_df)}, 验证集{len(val_df)}, 测试集{len(test_df)}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            # 创建示例数据用于测试
            return self._create_sample_data()
    
    def _add_text_content(self, df):
        """为数据框添加文本内容"""
        # 这里简化处理，实际应该从JSON文件加载
        df['text'] = df['id'].astype(str) + " 这是一条示例文本内容"
        return df
    
    def _create_sample_data(self):
        """创建示例数据"""
        sample_data = {
            'id': range(100),
            'stance': ['Against'] * 25 + ['Neither'] * 50 + ['Favor'] * 25,
            'intent': ['Political_interest'] * 20 + ['Economic_interest'] * 20 + 
                     ['Psychological_fulfillment'] * 20 + ['Public_interest'] * 20 +
                     ['Political_interest&Economic_interest'] * 10 + 
                     ['Psychological_fulfillment&Public_interest'] * 10,
            'fairness': ['Tinted'] * 50 + ['Fairness'] * 50,
            'harmfulness': ['Harmful'] * 60 + ['Unharmful'] * 40,
            'text': ['这是一条示例文本 ' + str(i) for i in range(100)]
        }
        
        df = pd.DataFrame(sample_data)
        train_df = df.iloc[:70]
        val_df = df.iloc[70:85]
        test_df = df.iloc[85:]
        
        return train_df, val_df, test_df

def compute_intent_class_weights(train_df):
    """计算多标签分类的类别权重 - 处理类别不平衡"""
    intent_labels = ['Political_interest', 'Economic_interest', 
                    'Psychological_fulfillment', 'Public_interest']
    
    # 统计每个标签的出现次数
    label_counts = [0] * len(intent_labels)
    total_samples = len(train_df)
    
    for idx, row in train_df.iterrows():
        intent_str = str(row['intent'])
        if pd.isna(intent_str) or intent_str == '':
            continue
            
        intent_list = intent_str.split('&')
        for i, label in enumerate(intent_labels):
            if label in intent_list:
                label_counts[i] += 1
    
    # 计算权重：使用逆频率加权
    weights = []
    for count in label_counts:
        if count > 0:
            # 使用平滑的逆频率加权
            weight = total_samples / (len(intent_labels) * count)
        else:
            weight = 1.0  # 如果某个标签没有出现，使用默认权重
        weights.append(weight)
    
    print(f"Intent标签分布: {dict(zip(intent_labels, label_counts))}")
    print(f"Intent类别权重: {dict(zip(intent_labels, [f'{w:.2f}' for w in weights]))}")
    
    return torch.tensor(weights, dtype=torch.float)

def predict_intent_with_strategy(model_output, strategy='adaptive', threshold=0.5, k=2):
    """
    多标签预测策略
    
    Args:
        model_output: 模型输出的logits
        strategy: 预测策略 ('adaptive', 'threshold', 'topk')
        threshold: 固定阈值
        k: top-k的k值
    """
    intent_probs = torch.sigmoid(model_output)
    
    if strategy == 'adaptive':
        # 自适应策略：基于平均标签数选择top-k
        avg_labels = max(1, int(intent_probs.sum(dim=1).float().mean().item()))
        intent_preds = torch.zeros_like(intent_probs)
        for i in range(intent_probs.size(0)):
            topk_indices = torch.topk(intent_probs[i], avg_labels).indices
            intent_preds[i, topk_indices] = 1
            
    elif strategy == 'threshold':
        # 固定阈值策略
        intent_preds = (intent_probs > threshold).float()
        
    elif strategy == 'topk':
        # 固定top-k策略
        intent_preds = torch.zeros_like(intent_probs)
        for i in range(intent_probs.size(0)):
            topk_indices = torch.topk(intent_probs[i], k).indices
            intent_preds[i, topk_indices] = 1
            
    else:
        raise ValueError(f"未知的预测策略: {strategy}")
    
    return intent_preds

class IntentMetrics:
    """意图分类评估指标"""
    @staticmethod
    def compute_metrics(predictions, labels):
        """计算多标签分类指标"""
        if len(predictions) == 0 or len(labels) == 0:
            return {
                'exact_match': 0.0,
                'macro_f1': 0.0,
                'micro_f1': 0.0,
                'hamming_loss': 1.0,
                'per_label_f1': [0.0, 0.0, 0.0, 0.0],
                'per_label_precision': [0.0, 0.0, 0.0, 0.0],
                'per_label_recall': [0.0, 0.0, 0.0, 0.0]
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
            
            # 汉明损失
            hamming_loss = np.mean(predictions_array != labels_array)
            
            # F1分数
            macro_f1 = f1_score(labels_array, predictions_array, average='macro', zero_division=0)
            micro_f1 = f1_score(labels_array, predictions_array, average='micro', zero_division=0)
            
            # 每个标签的指标
            per_label_f1 = f1_score(labels_array, predictions_array, average=None, zero_division=0)
            per_label_precision = precision_score(labels_array, predictions_array, average=None, zero_division=0)
            per_label_recall = recall_score(labels_array, predictions_array, average=None, zero_division=0)
            
            return {
                'exact_match': float(exact_match),
                'macro_f1': float(macro_f1),
                'micro_f1': float(micro_f1),
                'hamming_loss': float(hamming_loss),
                'per_label_f1': per_label_f1.tolist(),
                'per_label_precision': per_label_precision.tolist(),
                'per_label_recall': per_label_recall.tolist()
            }
            
        except Exception as e:
            print(f"计算指标时出错: {e}")
            return {
                'exact_match': 0.0,
                'macro_f1': 0.0,
                'micro_f1': 0.0,
                'hamming_loss': 1.0,
                'per_label_f1': [0.0, 0.0, 0.0, 0.0],
                'per_label_precision': [0.0, 0.0, 0.0, 0.0],
                'per_label_recall': [0.0, 0.0, 0.0, 0.0]
            }
    
    @staticmethod
    def print_detailed_report(predictions, labels, label_names):
        """打印详细的分类报告"""
        try:
            print("\n" + "="*60)
            print("详细分类报告")
            print("="*60)
            
            predictions_array = np.array(predictions)
            labels_array = np.array(labels)
            
            # 整体指标
            metrics = IntentMetrics.compute_metrics(predictions, labels)
            print(f"精确匹配: {metrics['exact_match']:.4f}")
            print(f"Macro F1: {metrics['macro_f1']:.4f}")
            print(f"Micro F1: {metrics['micro_f1']:.4f}")
            print(f"汉明损失: {metrics['hamming_loss']:.4f}")
            
            # 每个标签的指标
            print("\n每个标签的指标:")
            for i, label in enumerate(label_names):
                print(f"  {label}:")
                print(f"    F1: {metrics['per_label_f1'][i]:.4f}")
                print(f"    精确率: {metrics['per_label_precision'][i]:.4f}")
                print(f"    召回率: {metrics['per_label_recall'][i]:.4f}")
                
        except Exception as e:
            print(f"生成详细报告时出错: {e}")

class AdvancedIntentTrainer:
    """高级意图分类训练器"""
    def __init__(self, model, train_loader, val_loader, device, num_epochs=10, 
                 intent_class_weights=None, learning_rate=2e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 损失函数
        if intent_class_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=intent_class_weights.to(device))
            print(f"使用加权的多标签损失函数")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        # 训练状态
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        
        # 早停配置
        self.patience = 5
        self.counter = 0
        
        print(f"训练器初始化完成: {num_epochs}个epochs, 学习率{learning_rate}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_batches = len(self.train_loader)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            intent_labels = batch['intent'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss = self.criterion(outputs['intent_logits'], intent_labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化步骤
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # 进度显示
            if batch_idx % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                progress = (batch_idx / total_batches) * 100
                print(f'  Epoch {epoch}, Batch {batch_idx}/{total_batches} ({progress:.1f}%), '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.2e}')
        
        avg_loss = total_loss / total_batches
        epoch_time = time.time() - start_time
        
        return avg_loss, epoch_time
    
    def validate(self, strategy='adaptive'):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_batches = len(self.val_loader)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                intent_labels = batch['intent'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask)
                
                # 计算损失
                loss = self.criterion(outputs['intent_logits'], intent_labels)
                total_loss += loss.item()
                
                # 预测
                intent_preds = predict_intent_with_strategy(
                    outputs['intent_logits'], 
                    strategy=strategy
                )
                
                all_predictions.extend(intent_preds.cpu().numpy())
                all_labels.extend(intent_labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / total_batches
        metrics = IntentMetrics.compute_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def train(self):
        """完整训练过程"""
        print("开始意图分类训练...")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, epoch_time = self.train_epoch(epoch + 1)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_metrics['macro_f1'])
            
            # 打印结果
            print(f"\nEpoch {epoch+1} 结果:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"  验证Micro F1: {val_metrics['micro_f1']:.4f}")
            print(f"  Epoch时间: {epoch_time:.2f}s")
            
            # 保存最佳模型
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': self.best_val_f1,
                    'val_metrics': val_metrics
                }, 'outputs/best_intent_model.pth')
                print("✓ 保存最佳模型!")
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"⚠ 早停: {self.patience}个epoch F1分数未提升")
                    break
            
            print("=" * 60)
        
        # 训练总结
        print(f"\n训练完成! 最佳验证F1: {self.best_val_f1:.4f}")
        
        # 绘制训练曲线
        self._plot_training_curve()
    
    def _plot_training_curve(self):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            # 损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='训练损失')
            plt.plot(self.val_losses, label='验证损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('训练和验证损失')
            
            # F1分数曲线
            plt.subplot(1, 2, 2)
            plt.plot(self.val_f1_scores, label='验证Macro F1', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.title('验证F1分数')
            
            plt.tight_layout()
            plt.savefig('outputs/training_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("训练曲线已保存到 outputs/training_curve.png")
            
        except ImportError:
            print("Matplotlib未安装，跳过训练曲线绘制")

def main():
    """主函数"""
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # 1. 加载模型和tokenizer
        model_path = "C:/Users/Administrator/Desktop/check919/models/chinese-roberta-wwm-ext"
        print("加载BERT模型和tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_model = BertModel.from_pretrained(model_path)
        
        # 2. 加载数据
        print("加载数据...")
        data_processor = IntentDataProcessor()
        train_df, val_df, test_df = data_processor.load_data()
        
        # 3. 计算类别权重
        print("计算类别权重...")
        intent_class_weights = compute_intent_class_weights(train_df)
        
        # 4. 创建数据集和数据加载器
        print("创建数据集...")
        train_dataset = IntentDataset(train_df, tokenizer)
        val_dataset = IntentDataset(val_df, tokenizer)
        test_dataset = IntentDataset(test_df, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        print(f"数据加载器: 训练{len(train_loader)}批, 验证{len(val_loader)}批, 测试{len(test_loader)}批")
        
        # 5. 创建模型
        print("创建意图分类模型...")
        model = AdvancedIntentModel(bert_model, num_intents=4)
        model = model.to(device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数: 总计{total_params:,}, 可训练{trainable_params:,}")
        
        # 6. 创建训练器并开始训练
        print("开始训练...")
        trainer = AdvancedIntentTrainer(
            model, train_loader, val_loader, device,
            num_epochs=10,
            intent_class_weights=intent_class_weights,
            learning_rate=2e-5
        )
        
        trainer.train()
        
        # 7. 测试最佳模型
        print("\n测试最佳模型...")
        checkpoint = torch.load('outputs/best_intent_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        test_predictions = []
        test_labels = []
        gate_weights = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                # 预测
                intent_preds = predict_intent_with_strategy(outputs['intent_logits'])
                test_predictions.extend(intent_preds.cpu().numpy())
                test_labels.extend(intent_labels.cpu().numpy())
                gate_weights.extend(outputs['gate_weights'].cpu().numpy())
        
        # 计算测试指标
        test_metrics = IntentMetrics.compute_metrics(test_predictions, test_labels)
        
        # 打印详细报告
        IntentMetrics.print_detailed_report(test_predictions, test_labels, 
                                          ['Political', 'Economic', 'Psychological', 'Public'])
        
        # 分析门控权重
        if len(gate_weights) > 0:
            gate_weights = np.array(gate_weights)
            avg_gate_weights = np.mean(gate_weights, axis=0)
            print(f"\n平均门控权重:")
            print(f"  信念视角: {avg_gate_weights[0]:.4f}")
            print(f"  欲望视角: {avg_gate_weights[1]:.4f}")
            print(f"  计划视角: {avg_gate_weights[2]:.4f}")
        
        # 保存最终结果
        with open('outputs/final_results.txt', 'w', encoding='utf-8') as f:
            f.write("意图分类模型最终结果\n")
            f.write("="*50 + "\n")
            f.write(f"最佳验证F1: {checkpoint['val_f1']:.4f}\n")
            f.write(f"测试Macro F1: {test_metrics['macro_f1']:.4f}\n")
            f.write(f"测试Micro F1: {test_metrics['micro_f1']:.4f}\n")
            f.write(f"精确匹配: {test_metrics['exact_match']:.4f}\n")
            f.write(f"汉明损失: {test_metrics['hamming_loss']:.4f}\n")
            if len(gate_weights) > 0:
                f.write(f"平均门控权重 - 信念: {avg_gate_weights[0]:.4f}, 欲望: {avg_gate_weights[1]:.4f}, 计划: {avg_gate_weights[2]:.4f}\n")
        
        print(f"\n训练完成! 结果已保存到 outputs/final_results.txt")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 导入所需的sklearn指标
    from sklearn.metrics import precision_score, recall_score
    
    main()