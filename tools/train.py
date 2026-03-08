import argparse
import copy
import os
import sys
sys.path.append(os.getcwd())
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def _init_distributed():
    """初始化分布式训练，同时支持 torchpack/mpirun 和 torchrun 两种启动器。

    返回 True 表示已启用分布式，False 表示单卡模式。
    """
    # 方式 1: torchpack dist-run / mpirun (OpenMPI)
    #   特征: 设置 OMPI_COMM_WORLD_RANK 环境变量
    #   torchpack.distributed.init() 内部用 mpi4py 获取 rank，
    #   并读取 MASTER_HOST 做 torch.distributed.init_process_group
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        dist.init()
        return True

    # 方式 2: torchrun / torch.distributed.launch
    #   特征: 设置 RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    #   不走 mpi4py，直接用 torch.distributed 的 env:// 初始化
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
            )

        # 补丁 torchpack 模块变量，使 dist.rank() / dist.local_rank() 等仍可用
        import torchpack.distributed.context as _tp_ctx
        _tp_ctx._world_size = world_size
        _tp_ctx._world_rank = rank
        _tp_ctx._local_size = world_size  # 单节点假设
        _tp_ctx._local_rank = local_rank
        return True

    return False


def main():
    # ========== 初始化分布式训练 ==========
    distributed = _init_distributed()
    if not distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip validation during training (useful when sweeps data unavailable)")
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
        validate=not args.no_validate,  # 每个epoch进行验证（除非指定 --no-validate）
        timestamp=timestamp,  # 时间戳（用于标记实验）
    )

if __name__ == "__main__":
    main()
