import pandas as pd
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_curve
import numpy as np
from typing import Dict, List, Tuple
import time
from sklearn.utils.class_weight import compute_class_weight
import sys

# 1. 替换 Intent Loss：加权 BCE → ASL
class AsymmetricLoss(nn.Module):
    """ASL 损失函数 - 专门处理多标签不平衡"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        # 计算概率
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # 非对称裁剪
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 基础CE损失
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # 非对称权重
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

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
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        
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
    """差异化多视角特征提取器"""
    def __init__(self, perspective_type: str, input_dim: int = 768, output_dim: int = 256, dropout: float = 0.1):
        super(DifferentiatedFeatureExtractor, self).__init__()
        self.perspective_type = perspective_type
        
        if perspective_type == "belief":
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
            self.feature_net = nn.Sequential(
                nn.Linear(input_dim, 448),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(448, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:  # plan
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
    """增强的意图分类器 - 移除有问题的Attention层"""
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接返回logits，移除有问题的attention
        intent_logits = self.intent_net(x)
        return intent_logits

class GatedIntentAggregator(nn.Module):
    """门控意图聚合器"""
    def __init__(self, feature_dim: int = 256, num_intents: int = 4, dropout: float = 0.1):
        super(GatedIntentAggregator, self).__init__()
        
        # 门控机制
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 3),
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
    """增强版DMINT模型 - 优化版本"""
    def __init__(self, bert_model, hidden_size: int = 768, feature_dim: int = 256, dropout: float = 0.3):
        super(EnhancedDMINTModel, self).__init__()
        
        self.bert = bert_model
        
        # 多尺度特征提取器
        self.multi_scale_extractor = MultiScaleFeatureExtractor(hidden_size, feature_dim, dropout)
        
        # 三个差异化的视角特征提取器
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
            nn.Linear(128, 3)
        )
        
        self.harmfulness_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        self.fairness_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        
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
        multi_scale_features = self.dropout(multi_scale_features)
        
        # 门控意图聚合
        intent_output = self.intent_aggregator(belief_features, desire_features, plan_features)
        fused_features = intent_output['fused_features']
        intent_logits = intent_output['intent_logits']
        
        # 任务分类
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
            'gate_weights': intent_output['gate_weights']
        }

class OptimizedDMINTTrainer:
    """优化版训练器 - 修复所有关键问题"""
    def __init__(self, model, train_loader, val_loader, device, num_epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 损失函数 - 使用ASL替代加权BCE
        self.criterion_stance = nn.CrossEntropyLoss()
        self.criterion_intent = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05)  # 关键修复1
        self.criterion_harmfulness = nn.CrossEntropyLoss()
        self.criterion_fairness = nn.CrossEntropyLoss()
        
        # 3. 给 Intent 头单独加高 LR - 关键修复3
        intent_head_params = []
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'intent' in name.lower():
                    intent_head_params.append(param)
                else:
                    other_params.append(param)
        
        print(f"Intent头参数: {len(intent_head_params)}，其他参数: {len(other_params)}")
        
        self.optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': 2e-5},
            {'params': intent_head_params, 'lr': 5e-5}  # Intent头更高学习率
        ], weight_decay=0.01)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=2,
            factor=0.5
        )
        
        self.best_val_loss = float('inf')
        self.best_intent_f1 = 0.0
        self.patience = 5
        self.counter = 0
        
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
            
            # 计算多任务损失 - 移除固定权重，让ASL自己处理
            loss_stance = self.criterion_stance(outputs['stance'], stance_labels)
            loss_intent = self.criterion_intent(outputs['intent'], intent_labels)
            loss_harmfulness = self.criterion_harmfulness(outputs['harmfulness'], harmfulness_labels)
            loss_fairness = self.criterion_fairness(outputs['fairness'], fairness_labels)
            
            # 总损失 - 不再给intent固定4倍权重
            total_loss_batch = loss_stance + loss_intent + loss_harmfulness + loss_fairness
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
    
    def predict_intent(self, intent_logits, intent_labels=None):
        """2. 干掉top-k阈值，改用常数阈值 - 关键修复2"""
        intent_probs = torch.sigmoid(intent_logits)
        # 使用固定阈值0.3，而不是自适应top-k
        intent_preds = (intent_probs > 0.3).float()
        return intent_preds
    
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
                
                total_loss_batch = loss_stance + loss_intent + loss_harmfulness + loss_fairness
                total_loss += total_loss_batch.item()
                
                # 收集预测结果
                all_predictions['stance'].extend(torch.argmax(outputs['stance'], 1).cpu().numpy())
                all_labels['stance'].extend(stance_labels.cpu().numpy())
                
                all_predictions['harmfulness'].extend(torch.argmax(outputs['harmfulness'], 1).cpu().numpy())
                all_labels['harmfulness'].extend(harmfulness_labels.cpu().numpy())
                
                all_predictions['fairness'].extend(torch.argmax(outputs['fairness'], 1).cpu().numpy())
                all_labels['fairness'].extend(fairness_labels.cpu().numpy())
                
                # Intent多标签分类 - 使用固定阈值
                intent_preds = self.predict_intent(outputs['intent'])
                all_predictions['intent'].extend(intent_preds.cpu().numpy())
                all_labels['intent'].extend(intent_labels.cpu().numpy())
        
        # 计算指标
        metrics = self.compute_metrics(all_predictions, all_labels)
        avg_loss = total_loss / total_batches
        return avg_loss, metrics
    
    def compute_metrics(self, predictions, labels):
        """计算所有任务的指标"""
        metrics = {}
        
        # 单标签任务指标
        for task in ['stance', 'harmfulness', 'fairness']:
            if len(labels[task]) > 0:
                metrics[f'{task}_accuracy'] = accuracy_score(labels[task], predictions[task])
                metrics[f'{task}_f1'] = f1_score(labels[task], predictions[task], average='weighted')
            else:
                metrics[f'{task}_accuracy'] = 0.0
                metrics[f'{task}_f1'] = 0.0
        
        # Intent的多标签指标
        intent_metrics = self.safe_intent_metrics(predictions['intent'], labels['intent'])
        metrics['intent_exact_match'] = intent_metrics['exact_match']
        metrics['intent_macro_f1'] = intent_metrics['macro_f1']
        metrics['intent_micro_f1'] = intent_metrics['micro_f1']
        
        # 每个intent标签的F1
        intent_labels_names = ['Political', 'Economic', 'Psychological', 'Public']
        for i, label in enumerate(intent_labels_names):
            metrics[f'intent_{label}_f1'] = intent_metrics['per_label_f1'][i] if i < len(intent_metrics['per_label_f1']) else 0.0
        
        return metrics
    
    def safe_intent_metrics(self, predictions, labels):
        """安全地计算多标签分类指标"""
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
            
            if predictions_array.ndim == 1:
                predictions_array = predictions_array.reshape(1, -1)
                labels_array = labels_array.reshape(1, -1)
            
            exact_match = np.mean(np.all(predictions_array == labels_array, axis=1))
            
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
    
    def train(self):
        """完整训练过程"""
        print("开始优化训练...")
        total_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            train_loss, epoch_time = self.train_epoch(epoch + 1)
            val_loss, val_metrics = self.validate()
            
            self.scheduler.step(val_loss)
            
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
                torch.save(self.model.state_dict(), 'outputs/optimized_dmint_model.pth')
                print("✓ 保存最佳模型! (基于intent F1)")
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"⚠ 早停: {self.patience} 个epoch意图F1分数未提升")
                    break
            
            print("=" * 60)

# 数据加载函数保持不变
def load_and_preprocess_data():
    """加载和预处理数据"""
    data_dir = "data/"
    
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), sep=',', header=None,
                          names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"), sep=',', header=None,
                        names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
    test_df = pd.read_csv(os.path.join(data_dir, "test1.csv"), sep=',', header=None,
                         names=['id', 'stance', 'intent', 'fairness', 'harmfulness'])
    
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
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    # 4. 检查测试集分布 - 关键修复4
    print("\n标签分布检查:")
    print(f"训练集 Fairness: {train_df['fairness'].value_counts(normalize=True).to_dict()}")
    print(f"验证集 Fairness: {val_df['fairness'].value_counts(normalize=True).to_dict()}")
    print(f"测试集 Fairness: {test_df['fairness'].value_counts(normalize=True).to_dict()}")
    
    return train_df, val_df, test_df

def main_optimized_training():
    """优化版主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_path = "C:/Users/Administrator/Desktop/check919/models/chinese-roberta-wwm-ext"
    
    try:
        # 加载tokenizer和模型
        print("加载模型...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_model = BertModel.from_pretrained(model_path)
        print("模型加载成功!")
        
        # 加载数据
        train_df, val_df, test_df = load_and_preprocess_data()
        
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
        
        # 创建优化版DMINT模型
        model = EnhancedDMINTModel(bert_model)
        model = model.to(device)
        print("优化版DMINT模型创建成功!")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 确保输出目录存在
        os.makedirs('outputs', exist_ok=True)
        
        # 使用优化版训练器
        trainer = OptimizedDMINTTrainer(
            model, train_loader, val_loader, device, num_epochs=5
        )
        trainer.train()
        
        # 测试最佳模型
        print("\n加载最佳模型进行测试...")
        model.load_state_dict(torch.load('outputs/optimized_dmint_model.pth'))
        
        # 测试代码（与训练器中的validate类似）
        model.eval()
        test_predictions = {'stance': [], 'intent': [], 'harmfulness': [], 'fairness': []}
        test_labels = {'stance': [], 'intent': [], 'harmfulness': [], 'fairness': []}
        
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
                
                # 使用相同的固定阈值
                intent_preds = trainer.predict_intent(outputs['intent'])
                test_predictions['intent'].extend(intent_preds.cpu().numpy())
                test_labels['intent'].extend(intent_labels.cpu().numpy())
        
        # 计算测试指标
        test_metrics = trainer.compute_metrics(test_predictions, test_labels)
        
        print("\n" + "="*50)
        print("优化版测试集结果:")
        print("="*50)
        
        for task in ['stance', 'harmfulness', 'fairness']:
            acc = test_metrics[f'{task}_accuracy']
            f1 = test_metrics[f'{task}_f1']
            print(f"{task}: 准确率={acc:.4f}, F1={f1:.4f}")
        
        print(f"intent: 精确匹配={test_metrics['intent_exact_match']:.4f}, "
              f"F1-macro={test_metrics['intent_macro_f1']:.4f}, "
              f"F1-micro={test_metrics['intent_micro_f1']:.4f}")
        
        # 保存结果
        with open('outputs/optimized_final_results.txt', 'w', encoding='utf-8') as f:
            f.write("优化版DMINT模型最终测试结果\n")
            f.write("="*50 + "\n")
            for task in ['stance', 'harmfulness', 'fairness']:
                acc = test_metrics[f'{task}_accuracy']
                f1 = test_metrics[f'{task}_f1']
                f.write(f"{task}: 准确率={acc:.4f}, F1={f1:.4f}\n")
            f.write(f"intent: 精确匹配={test_metrics['intent_exact_match']:.4f}, "
                   f"F1-macro={test_metrics['intent_macro_f1']:.4f}, "
                   f"F1-micro={test_metrics['intent_micro_f1']:.4f}\n")
        
        print("\n优化训练完成! 结果已保存到 outputs/optimized_final_results.txt")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_optimized_training()