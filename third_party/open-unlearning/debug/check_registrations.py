#!/usr/bin/env python3
"""检查数据集是否正确注册"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_data_registrations():
    """检查数据集注册情况"""
    print("=== Checking Data Registrations ===")
    
    try:
        from data import _DATASET_REGISTRY
        print("Available datasets:")
        for name, handler in _DATASET_REGISTRY.items():
            print(f"  - {name}: {handler}")
        
        # 检查我们需要的数据集是否注册
        required_datasets = ["CelebrityUnlearnDataset", "WikiTextRetainDataset"]
        for dataset_name in required_datasets:
            if dataset_name in _DATASET_REGISTRY:
                print(f"✅ {dataset_name} is registered")
            else:
                print(f"❌ {dataset_name} is NOT registered")
                
    except Exception as e:
        print(f"❌ Failed to check registrations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data_registrations()
