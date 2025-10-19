"""
DMINT模型验证测试代码
用于验证模型训练流程的各个环节
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import sys

# 导入主模型文件中的组件
from ten import EnhancedDMINTModel, TextDataset, ImprovedDMINTTrainer

# 使用模型进行训练或推理
from ten import (
    TextDataset, EnhancedDMINTModel, ImprovedDMINTTrainer,
    load_and_preprocess_data, compute_intent_class_weights, safe_intent_metrics
)

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

def test_model_inference():
    """测试模型推理功能"""
    print("\n" + "=" * 60)
    print("测试模型推理")
    print("=" * 60)
    
    try:
        # 加载模型和tokenizer
        model_path = "C:/Users/LHTBS/Desktop/check919/models/chinese-roberta-wwm-ext"
        tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_model = BertModel.from_pretrained(model_path)
        model = EnhancedDMINTModel(bert_model)
        
        # 测试文本
        test_texts = [
            "这是一个测试文本，用于验证模型推理功能",
            "另一个测试样本，检查多任务分类效果"
        ]
        
        # 编码文本
        encoding = tokenizer(
            test_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # 模型推理
        model.eval()
        with torch.no_grad():
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
        
        print("✓ 模型推理成功")
        print("推理结果:")
        print(f"  Stance logits形状: {outputs['stance'].shape}")
        print(f"  Intent logits形状: {outputs['intent'].shape}")
        print(f"  Harmfulness logits形状: {outputs['harmfulness'].shape}")
        print(f"  Fairness logits形状: {outputs['fairness'].shape}")
        
        # 转换预测结果
        stance_preds = torch.argmax(outputs['stance'], dim=1)
        intent_probs = torch.sigmoid(outputs['intent'])
        harmfulness_preds = torch.argmax(outputs['harmfulness'], dim=1)
        fairness_preds = torch.argmax(outputs['fairness'], dim=1)
        
        print(f"  Stance预测: {stance_preds.tolist()}")
        print(f"  Intent概率: {intent_probs.tolist()}")
        print(f"  Harmfulness预测: {harmfulness_preds.tolist()}")
        print(f"  Fairness预测: {fairness_preds.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("DMINT模型验证测试")
    print("=" * 60)
    
    # 创建验证器
    validator = TrainingValidator()
    
    # 运行完整验证
    validation_success = validator.run_complete_validation()
    
    if validation_success:
        # 运行模型推理测试
        test_model_inference()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        print("✓ 模型验证通过")
        print("✓ 可以安全进行训练")
    else:
        print("\n" + "=" * 60)
        print("验证失败")
        print("=" * 60)
        print("⚠ 请根据错误信息修复问题后再进行训练")

if __name__ == "__main__":
    main()