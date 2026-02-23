import argparse
import copy
import os
import sys
sys.path.append(os.getcwd())
import random
import time  # 时间模块（记录时间戳）

import numpy as np
import torch
from mmcv import Config  # MMCV配置管理
from torchpack import distributed as dist  # 分布式训练（多GPU）
from torchpack.environ import auto_set_run_dir, set_run_dir  # 自动设置运行目录
from torchpack.utils.config import configs  # 配置加载工具

from mmdet3d.apis import train_model  # 3D目标检测训练函数
from mmdet3d.datasets import build_dataset  # 数据集构建器
from mmdet3d.models import build_model  # 模型构建器
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def main():
    # ========== 初始化分布式训练 ==========
    # 设置多GPU分布式训练环境（torchpack框架）
    # 检测环境变量RANK和WORLD_SIZE，如果有则启用分布式模式
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init()
        distributed = True
    else:
        distributed = False
        # 单机模式下默认使用第一块GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser()
    # 位置参数：配置文件路径（必需）
    parser.add_argument("config", metavar="FILE", help="config file")
    # 可选参数：运行输出目录
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    # args: 主要参数，opts: 其他选项（如学习率等）
    args, opts = parser.parse_known_args()

    # ========== 配置加载与合并 ==========
    # 从配置文件递归加载所有配置
    configs.load(args.config, recursive=True)
    # 用命令行参数覆盖配置文件中的值
    # 例如：--lr=0.001 会覆盖配置文件中的学习率
    configs.update(opts)

    # 将配置转换为MMCV Config对象（便于访问）
    cfg = Config(recursive_eval(configs), filename=args.config)
    
    if not distributed:
        cfg.gpu_ids = [0]
    else:
        cfg.gpu_ids = range(1)


    # ========== 分布式GPU配置 ==========
    # 设置cuDNN基准（决定是否使用不确定性加速）
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    # 指定当前进程使用的GPU设备（每个GPU进程分配一个）
    if distributed:
        torch.cuda.set_device(dist.local_rank())

    # ========== 运行目录设置 ==========
    # 如果命令行没有指定目录，自动生成唯一目录名（避免覆盖）
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        # 否则使用指定的目录
        set_run_dir(args.run_dir)
    # 将运行目录保存到配置对象
    cfg.run_dir = args.run_dir

    # ========== 配置文件导出 ==========
    # 将最终的配置保存为YAML文件，便于查阅和复现
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # ========== 日志初始化 ==========
    # 生成时间戳用于日志文件名（格式：年月日_时分秒）
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # 日志文件路径：运行目录/时间戳.log
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    # 初始化根日志记录器（同时输出到控制台和文件）
    logger = get_root_logger(log_file=log_file)

    # ========== 日志输出基本信息 ==========
    # 打印完整的配置信息（便于调试和记录实验设置）
    # logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Config:\n{cfg}")

    # ========== 随机种子设置 ==========
    # 设置随机种子确保实验可复现
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        # Python随机数生成器
        random.seed(cfg.seed)
        # NumPy随机数生成器
        np.random.seed(cfg.seed)
        # PyTorch CPU随机数生成器
        torch.manual_seed(cfg.seed)
        
        # 如果启用确定性模式，强制使用确定性算法（性能可能降低）
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ========== 数据集构建 ==========
    # 根据配置文件构建训练数据集
    # cfg.data.train 包含数据集类型、路径、数据处理管道等信息
    datasets = [build_dataset(cfg.data.train)]
    
    # ========== 【断点1】数据集检查 ==========
    # 在这里放断点来检查数据集的基本信息
    dataset = datasets[0]
    # 在IDE中添加以下调试代码（可选）：
    # - 查看 dataset.data_infos 的长度（数据集大小）
    # - 查看 dataset.CLASSES 类别列表
    print(f"DEBUG: Dataset size: {len(dataset)}")
    print(f"DEBUG: Dataset classes: {dataset.CLASSES}")
    if len(dataset) > 0:
        # 获取第一个样本查看数据结构
        sample_idx = 0
        sample_data = dataset[sample_idx]
        print(f"DEBUG: First sample keys: {sample_data.keys()}")
        for key in sample_data.keys():
            if isinstance(sample_data[key], torch.Tensor):
                print(f"  {key}: shape={sample_data[key].shape}, dtype={sample_data[key].dtype}")
            elif isinstance(sample_data[key], dict):
                print(f"  {key}: dict with keys {sample_data[key].keys()}")
            elif isinstance(sample_data[key], list):
                print(f"  {key}: list with {len(sample_data[key])} items")
            else:
                print(f"  {key}: {type(sample_data[key])}")

    # ========== 模型构建 ==========
    # 根据配置文件构建模型
    # cfg.model 包含模型架构、层数、输出通道数等信息
    model = build_model(cfg.model)
    # 初始化模型权重（使用随机初始化或预训练权重）
    model.init_weights()
    
    # ========== 同步批量归一化（可选） ==========
    # 多GPU训练时，BatchNorm需要同步各GPU的统计信息，以获得更稳定的结果
    if cfg.get("sync_bn", None):
        # 如果sync_bn不是字典，转换为字典格式
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        # 将模型中的BatchNorm层转换为SyncBatchNorm
        # exclude 参数用于排除某些不需要同步的层
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    # ========== 日志输出模型信息 ==========
    # 打印模型结构（便于检查是否正确）
    logger.info(f"Model:\n{model}")
    
    # ========== 开始训练 ==========
    # 调用训练函数执行完整的训练流程
    train_model(
        model,  # 待训练的模型
        datasets,  # 训练数据集列表
        cfg,  # 配置对象
        distributed=distributed,  # 启用分布式训练
        validate=True,  # 每个epoch进行验证
        timestamp=timestamp,  # 时间戳（用于标记实验）
    )

if __name__ == "__main__":
    main()
