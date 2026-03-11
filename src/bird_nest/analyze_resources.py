#!/usr/bin/env python3
"""
系统资源分析脚本 - 用于优化训练参数
分析 CPU、内存、GPU 等资源，给出推荐的训练参数
"""

import os
import sys
import multiprocessing
import platform
import psutil
import torch

def analyze_system_resources():
    """分析系统资源"""
    print("=" * 60)
    print("系统资源分析")
    print("=" * 60)
    
    # CPU 信息
    cpu_count = multiprocessing.cpu_count()
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"\n📊 CPU 信息:")
    print(f"  逻辑核心数: {cpu_count}")
    print(f"  物理核心数: {cpu_count_physical}")
    if cpu_freq:
        print(f"  频率: {cpu_freq.current:.0f} MHz (最大: {cpu_freq.max:.0f} MHz)")
    print(f"  当前使用率: {cpu_percent:.1f}%")
    
    # 内存信息
    mem = psutil.virtual_memory()
    mem_total_gb = mem.total / (1024**3)
    mem_available_gb = mem.available / (1024**3)
    mem_used_gb = mem.used / (1024**3)
    mem_percent = mem.percent
    
    print(f"\n💾 内存信息:")
    print(f"  总内存: {mem_total_gb:.2f} GB")
    print(f"  可用内存: {mem_available_gb:.2f} GB")
    print(f"  已用内存: {mem_used_gb:.2f} GB ({mem_percent:.1f}%)")
    
    # GPU 信息
    gpu_available = torch.cuda.is_available()
    print(f"\n🎮 GPU 信息:")
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        print(f"  CUDA 可用: ✅")
        print(f"  GPU 数量: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    显存: {props.total_memory / (1024**3):.2f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")
        
        # 当前 GPU 内存使用
        if gpu_count > 0:
            torch.cuda.set_device(0)
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n  GPU 0 内存使用:")
            print(f"    已分配: {gpu_mem_allocated:.2f} GB")
            print(f"    已保留: {gpu_mem_reserved:.2f} GB")
            print(f"    总计: {gpu_mem_total:.2f} GB")
            print(f"    可用: {gpu_mem_total - gpu_mem_reserved:.2f} GB")
    else:
        print(f"  CUDA 可用: ❌")
    
    # WSL 检测
    is_wsl = False
    if platform.system() == "Linux":
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower() or "wsl" in f.read().lower():
                    is_wsl = True
        except:
            pass
    
    print(f"\n🖥️  系统信息:")
    print(f"  操作系统: {platform.system()}")
    print(f"  是否 WSL: {'是' if is_wsl else '否'}")
    
    return {
        'cpu_count': cpu_count,
        'cpu_count_physical': cpu_count_physical,
        'mem_total_gb': mem_total_gb,
        'mem_available_gb': mem_available_gb,
        'mem_percent': mem_percent,
        'gpu_available': gpu_available,
        'gpu_count': torch.cuda.device_count() if gpu_available else 0,
        'gpu_mem_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_available else 0,
        'is_wsl': is_wsl,
    }

def recommend_parameters(resources):
    """根据资源分析推荐参数"""
    print("\n" + "=" * 60)
    print("参数推荐")
    print("=" * 60)
    
    cpu_count = resources['cpu_count']
    mem_available_gb = resources['mem_available_gb']
    mem_total_gb = resources['mem_total_gb']
    is_wsl = resources['is_wsl']
    gpu_available = resources['gpu_available']
    gpu_mem_total_gb = resources['gpu_mem_total_gb']
    
    # 1. num_workers 推荐
    print(f"\n📦 DataLoader num_workers:")
    
    # 基础计算
    base_workers = max(1, cpu_count // 4)  # 保守：1/4 CPU核心
    
    if is_wsl:
        # WSL 更保守
        recommended_workers = min(4, max(1, cpu_count // 4))
        print(f"  ⚠️  WSL 环境检测到，使用更保守的设置")
    else:
        recommended_workers = min(6, max(2, cpu_count // 3))
    
    # 根据内存调整
    if mem_available_gb < 4:
        recommended_workers = min(recommended_workers, 2)
        print(f"  ⚠️  可用内存较少 ({mem_available_gb:.1f} GB)，减少 workers")
    elif mem_available_gb < 8:
        recommended_workers = min(recommended_workers, 3)
        print(f"  ⚠️  可用内存中等 ({mem_available_gb:.1f} GB)，适度减少 workers")
    
    print(f"  推荐值: {recommended_workers}")
    print(f"  范围: 1-{min(8, cpu_count // 2)} (根据 CPU 核心数)")
    
    # 2. batch_size 推荐
    print(f"\n📊 Batch Size:")
    if gpu_available:
        if gpu_mem_total_gb >= 8:
            recommended_bs = 6
            bs_range = "4-8"
        elif gpu_mem_total_gb >= 6:
            recommended_bs = 4
            bs_range = "2-6"
        else:
            recommended_bs = 2
            bs_range = "1-4"
        print(f"  GPU 显存: {gpu_mem_total_gb:.1f} GB")
    else:
        recommended_bs = 2
        bs_range = "1-2"
        print(f"  CPU 模式")
    
    print(f"  推荐值: {recommended_bs}")
    print(f"  范围: {bs_range}")
    
    # 3. prefetch_factor 推荐
    print(f"\n⚡ Prefetch Factor:")
    if mem_available_gb < 4:
        recommended_prefetch = 1
    elif mem_available_gb < 8:
        recommended_prefetch = 2
    else:
        recommended_prefetch = 2 if is_wsl else 3
    
    print(f"  推荐值: {recommended_prefetch}")
    print(f"  说明: 每个 worker 预取的批次数量")
    if is_wsl:
        print(f"  ⚠️  WSL 环境，建议使用较小值 (1-2)")
    
    # 4. 内存估算
    print(f"\n💡 内存使用估算:")
    # 每个 worker 大约占用 200-500 MB
    worker_mem_mb = recommended_workers * 300
    # 每个 batch 大约占用 100-200 MB (取决于图像大小)
    batch_mem_mb = recommended_bs * 150
    # prefetch 额外内存
    prefetch_mem_mb = recommended_workers * recommended_prefetch * 100
    
    total_estimated_mb = worker_mem_mb + batch_mem_mb + prefetch_mem_mb
    total_estimated_gb = total_estimated_mb / 1024
    
    print(f"  Workers 内存: ~{worker_mem_mb / 1024:.1f} GB ({recommended_workers} workers)")
    print(f"  Batch 内存: ~{batch_mem_mb / 1024:.1f} GB (batch_size={recommended_bs})")
    print(f"  Prefetch 内存: ~{prefetch_mem_mb / 1024:.1f} GB")
    print(f"  总计估算: ~{total_estimated_gb:.1f} GB")
    print(f"  可用内存: {mem_available_gb:.1f} GB")
    
    if total_estimated_gb > mem_available_gb * 0.8:
        print(f"  ⚠️  警告: 估算内存使用可能超过可用内存的 80%")
        print(f"  建议: 减少 num_workers 或 batch_size")
    elif total_estimated_gb > mem_available_gb * 0.6:
        print(f"  ⚠️  注意: 估算内存使用较高，建议监控内存使用")
    else:
        print(f"  ✅ 内存使用在安全范围内")
    
    # 5. 生成配置建议
    print(f"\n" + "=" * 60)
    print("推荐配置")
    print("=" * 60)
    print(f"\n在 train_bird_nest.py 中设置:")
    print(f"  bs = {recommended_bs}")
    print(f"\n环境变量设置:")
    print(f"  export NUM_WORKERS={recommended_workers}")
    print(f"\n或修改代码中的 data_loader_params:")
    print(f"  'num_workers': {recommended_workers},")
    print(f"  'prefetch_factor': {recommended_prefetch},")
    
    return {
        'num_workers': recommended_workers,
        'batch_size': recommended_bs,
        'prefetch_factor': recommended_prefetch,
    }

def main():
    try:
        resources = analyze_system_resources()
        recommendations = recommend_parameters(resources)
        
        print(f"\n" + "=" * 60)
        print("使用建议")
        print("=" * 60)
        print(f"\n1. 运行训练时设置环境变量:")
        print(f"   NUM_WORKERS={recommendations['num_workers']} python src/bird_nest/train_bird_nest.py")
        print(f"\n2. 如果遇到内存不足，逐步减少:")
        print(f"   - 先减少 num_workers (减少 1-2)")
        print(f"   - 再减少 batch_size (减少 1)")
        print(f"   - 最后减少 prefetch_factor (减少到 1)")
        print(f"\n3. 监控系统资源:")
        print(f"   - 使用 'htop' 或 'top' 监控 CPU 和内存")
        print(f"   - 使用 'nvidia-smi' 监控 GPU")
        
    except Exception as e:
        print(f"\n❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

