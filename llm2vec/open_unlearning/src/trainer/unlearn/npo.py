from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff
import torch

class NPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        print("\n" + "="*50)
        print("NPO DEBUG: 开始调试数据结构")
        print("="*50)
        
        forget_inputs = inputs["forget"]
        print(f"1. forget_inputs 类型: {type(forget_inputs)}")
        
        if isinstance(forget_inputs, dict):
            print(f"2. forget_inputs 包含的键: {list(forget_inputs.keys())}")
            
            for key, value in forget_inputs.items():
                print(f"\n3. 检查键 '{key}':")
                print(f"   - 类型: {type(value)}")
                
                if isinstance(value, torch.Tensor):
                    print(f"   - 形状: {value.shape}")
                    print(f"   - 数据类型: {value.dtype}")
                    print(f"   - 设备: {value.device}")
                    print(f"   - 最小值: {torch.min(value).item()}")
                    print(f"   - 最大值: {torch.max(value).item()}")
                    
                    # 检查vocab_size
                    vocab_size = model.config.vocab_size
                    print(f"   - 模型vocab_size: {vocab_size}")
                    
                    if torch.any(value >= vocab_size):
                        invalid_count = torch.sum(value >= vocab_size).item()
                        invalid_indices = torch.where(value >= vocab_size)
                        print(f"   - ❌ 发现 {invalid_count} 个无效token ID (>= {vocab_size})")
                        print(f"   - 无效位置: {invalid_indices}")
                        
                    if torch.any(value < 0):
                        neg_count = torch.sum(value < 0).item()
                        print(f"   - ❌ 发现 {neg_count} 个负数token ID")
                        
                elif isinstance(value, dict):
                    print(f"   - 这是嵌套字典，包含键: {list(value.keys())}")
                    for nested_key, nested_value in value.items():
                        print(f"     * {nested_key}: {type(nested_value)}")
                        if isinstance(nested_value, torch.Tensor):
                            print(f"       形状: {nested_value.shape}, 最大值: {torch.max(nested_value).item()}")
                else:
                    print(f"   - 值: {value}")
        
        print("="*50)
        print("NPO DEBUG: 数据结构检查完毕")
        print("="*50 + "\n")

        forget_inputs = inputs["forget"]

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
