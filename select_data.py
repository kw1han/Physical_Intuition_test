import os
import random
from pathlib import Path
import argparse
import pandas as pd
import shutil
from collections import defaultdict

def analyze_trials(data_root):
    """
    分析数据集中所有试验，按游戏类型和成功/失败状态进行分类
    
    Args:
        data_root: 数据根目录
    
    Returns:
        dict: 游戏类型 -> (成功试验列表，失败试验列表) 的映射
    """
    game_type_trials = defaultdict(lambda: ([], []))
    
    # 遍历所有subject文件夹
    for subj_folder in Path(data_root).glob('Subj_*'):
        if not subj_folder.is_dir():
            continue
        
        # 遍历subject中的所有试验
        for trial_folder in subj_folder.iterdir():
            if not trial_folder.is_dir():
                continue
            
            # 解析文件夹名称
            parts = trial_folder.name.split('_')
            if len(parts) < 5:
                continue
            
            game_type = parts[0]
            is_success = trial_folder.name.endswith("True")
            
            # 按游戏类型和成功/失败状态分类
            success_trials, failure_trials = game_type_trials[game_type]
            if is_success:
                success_trials.append(trial_folder)
            else:
                failure_trials.append(trial_folder)
            
            game_type_trials[game_type] = (success_trials, failure_trials)
    
    return game_type_trials

def filter_balanced_dataset(data_root, output_dir, success_ratio=0.25):
    """
    筛选平衡的数据集，确保所有游戏类型的数量一致，且成功/失败比例控制在指定值
    
    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        success_ratio: 成功案例的比例，默认为0.25 (1:3)
    """
    print(f"分析数据集: {data_root}")
    game_type_trials = analyze_trials(data_root)
    
    # 统计各游戏类型的试验数量
    game_type_stats = {}
    for game_type, (success_trials, failure_trials) in game_type_trials.items():
        total = len(success_trials) + len(failure_trials)
        success_ratio_current = len(success_trials) / total if total > 0 else 0
        game_type_stats[game_type] = {
            'total': total,
            'success': len(success_trials),
            'failure': len(failure_trials),
            'success_ratio': success_ratio_current
        }
    
    # 打印初始统计数据
    print("\n原始数据集统计:")
    stats_df = pd.DataFrame.from_dict(game_type_stats, orient='index')
    stats_df = stats_df.sort_values('total', ascending=False)
    print(stats_df)
    
    # 计算每种游戏类型可以保留的最大平衡数量
    balanced_counts = {}
    for game_type, stats in game_type_stats.items():
        # 根据成功率限制，计算可以保留的最大样本数
        max_samples_by_success = int(stats['success'] / success_ratio) if success_ratio > 0 else float('inf')
        max_samples_by_failure = int(stats['failure'] / (1 - success_ratio)) if success_ratio < 1 else float('inf')
        max_balanced_samples = min(max_samples_by_success, max_samples_by_failure)
        balanced_counts[game_type] = max_balanced_samples
    
    # 找出所有游戏类型可以保留的公共最大数量
    min_count_per_type = min(balanced_counts.values()) if balanced_counts else 0
    
    # 如果最小数量太小，可以考虑调整成功率要求
    if min_count_per_type < 10:
        print(f"\n警告: 使用当前成功率{success_ratio}，每种游戏类型最多只能保留{min_count_per_type}个样本")
        print("考虑调整成功率要求或接受不平衡的游戏类型分布")
    
    # 计算每种游戏类型要保留的成功和失败样本数
    success_count_per_type = int(min_count_per_type * success_ratio)
    failure_count_per_type = min_count_per_type - success_count_per_type
    
    print(f"\n平衡后每种游戏类型将保留: {min_count_per_type}个样本")
    print(f"  - 成功样本: {success_count_per_type}")
    print(f"  - 失败样本: {failure_count_per_type}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 为每种游戏类型选择相应数量的样本
    selected_trials = []
    selected_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
    
    for game_type, (success_trials, failure_trials) in game_type_trials.items():
        # 随机选择指定数量的成功和失败样本
        selected_success = random.sample(success_trials, min(success_count_per_type, len(success_trials)))
        selected_failure = random.sample(failure_trials, min(failure_count_per_type, len(failure_trials)))
        
        # 更新统计信息
        selected_stats[game_type]['success'] = len(selected_success)
        selected_stats[game_type]['failure'] = len(selected_failure)
        
        # 添加到选中列表
        selected_trials.extend(selected_success)
        selected_trials.extend(selected_failure)
    
    # 打印选择结果
    print("\n选择的样本统计:")
    selected_df = pd.DataFrame({
        game_type: {
            'success': stats['success'],
            'failure': stats['failure'],
            'total': stats['success'] + stats['failure'],
            'success_ratio': stats['success'] / (stats['success'] + stats['failure']) if (stats['success'] + stats['failure']) > 0 else 0
        } for game_type, stats in selected_stats.items()
    }).T
    
    selected_df = selected_df.sort_values('total', ascending=False)
    print(selected_df)
    
    # 将选择的试验拷贝到输出目录
    print(f"\n复制选中的{len(selected_trials)}个试验到{output_dir}...")
    
    # 创建新的目录结构
    for i, trial_path in enumerate(selected_trials):
        # 保持原始目录结构
        subj_name = trial_path.parent.name
        trial_name = trial_path.name
        
        # 创建subject目录
        subj_output_dir = output_path / subj_name
        subj_output_dir.mkdir(exist_ok=True)
        
        # 拷贝试验文件夹
        trial_output_dir = subj_output_dir / trial_name
        
        if trial_output_dir.exists():
            continue  # 跳过已存在的目录
            
        # 拷贝文件夹
        shutil.copytree(trial_path, trial_output_dir)
        
        # 显示进度
        if (i + 1) % 10 == 0 or i + 1 == len(selected_trials):
            print(f"  进度: {i+1}/{len(selected_trials)} ({(i+1)/len(selected_trials)*100:.1f}%)")
    
    # 创建汇总文件
    with open(output_path / "dataset_summary.txt", "w") as f:
        f.write("=== 平衡数据集汇总 ===\n\n")
        f.write(f"原始数据路径: {data_root}\n")
        f.write(f"总样本数: {len(selected_trials)}\n")
        f.write(f"游戏类型数: {len(selected_stats)}\n")
        f.write(f"每种类型样本数: {min_count_per_type}\n")
        f.write(f"成功样本比例: {success_ratio:.2f}\n\n")
        
        f.write("游戏类型统计:\n")
        f.write(selected_df.to_string())
        
        f.write("\n\n选中的试验清单:\n")
        for trial in selected_trials:
            f.write(f"{trial}\n")
    
    print(f"\n完成! 平衡数据集已保存到 {output_dir}")
    print(f"汇总信息已保存到 {output_path / 'dataset_summary.txt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='创建平衡的物理直觉数据集')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--success_ratio', type=float, default=0.25, help='成功样本的比例 (默认: 0.25，即1:3)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于可重复性')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 运行筛选
    filter_balanced_dataset(args.data_root, args.output_dir, args.success_ratio)
