import argparse
import copy
import os
import sys
sys.path.append(os.getcwd())
import warnings
import numpy as np

# ========== NumPy版本兼容性处理 ==========
# 处理NumPy 1.20+中移除的别名（np.int, np.float等）
try:
    np.long = int
    np.int = int
    np.float = float
    np.bool = bool
except:
    pass

# ========== MMCV和PyTorch相关 ==========
import mmcv  # MMCV工具库
import torch  # PyTorch深度学习框架
from torchpack.utils.config import configs  # Torchpack配置工具
from torchpack import distributed as dist  # 分布式训练

# ========== 配置和工具 ==========
from mmcv import Config, DictAction  # 配置管理和字典操作
from mmcv.cnn import fuse_conv_bn  # 融合卷积和批量归一化
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel  # 并行训练
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model  # 运行器相关

# ========== MMDet3D API ==========
from mmdet3d.apis import single_gpu_test  # 单GPU测试
from mmdet3d.datasets import build_dataloader, build_dataset  # 数据集构建
from mmdet3d.models import build_model  # 模型构建
from mmdet.apis import multi_gpu_test, set_random_seed  # 多GPU测试和随机种子
from mmdet.datasets import replace_ImageToTensor  # 图像张量替换
from mmdet3d.utils import recursive_eval  # 递归求值


# ========== 命令行参数解析函数 ==========
def parse_args():
    """解析测试脚本的命令行参数"""
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    
    # ========== 必需参数 ==========
    # 测试配置文件路径
    parser.add_argument("config", help="test config file path")
    # 模型检查点文件（保存的训练权重）
    parser.add_argument("checkpoint", help="checkpoint file")
    
    # ========== 输出相关参数 ==========
    # 输出结果保存路径（pickle格式）
    parser.add_argument("--out", help="output result file in pickle format")
    # 融合卷积和批量归一化层以加速推理
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    # 仅格式化输出，不进行评估（用于提交到测试服务器）
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    
    # ========== 评估参数 ==========
    # 评估指标（如mAP、recall等，根据数据集而定）
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    
    # ========== 可视化参数 ==========
    # 显示预测结果（实时可视化）
    parser.add_argument("--show", action="store_true", help="show results")
    # 保存结果可视化图像的目录
    parser.add_argument("--show-dir", help="directory where results will be saved")
    
    # ========== 多GPU测试参数 ==========
    # 使用GPU收集多GPU测试的结果
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    # 多GPU结果收集的临时目录
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    
    # ========== 可复现性参数 ==========
    # 随机种子
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # 使用确定性算法
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    
    # ========== 配置覆盖参数 ==========
    # 通过命令行覆盖配置文件中的参数
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    
    # ========== 评估选项（旧参数，已弃用） ==========
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    
    # ========== 评估选项（新参数） ==========
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    
    # ========== 分布式训练参数 ==========
    # 任务启动器选择（用于多机多GPU）
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    # 本地GPU_rank（用于分布式）
    parser.add_argument("--local_rank", type=int, default=0)
    
    # 解析参数
    args = parser.parse_args()
    
    # ========== 环境变量设置 ==========
    # 设置LOCAL_RANK环境变量（用于分布式训练）
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    # ========== 参数检查和兼容性处理 ==========
    # 检查--options和--eval-options是否同时指定（不允许）
    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    # 如果使用旧参数--options，则转换为--eval-options
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    
    return args


# ========== 主测试函数 ==========
def main():
    """执行模型测试和评估的主函数"""
    # 解析命令行参数
    args = parse_args()
    # dist.init()  # 注释：分布式测试需要取消注释

    # ========== GPU配置 ==========
    # 启用cuDNN自动优化以加快推理速度
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(dist.local_rank())  # 注释：分布式测试需要取消注释

    # ========== 参数验证 ==========
    # 确保至少指定一种操作（保存/评估/格式化/显示）
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    # --eval和--format-only不能同时指定
    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    # 输出文件必须是pickle格式
    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    # ========== 配置加载 ==========
    # 加载测试配置文件
    configs.load(args.config, recursive=True)
    # 转换为MMCV Config对象
    cfg = Config(recursive_eval(configs), filename=args.config)
    print(cfg)

    # ========== 配置覆盖 ==========
    # 用命令行参数覆盖配置文件中的值
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # ========== 测试数据集配置 ==========
    # 禁用预训练模型（使用检查点中的权重）
    cfg.model.pretrained = None
    # 处理测试数据集配置（单个数据集或多个数据集）
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        # 单个数据集情况
        cfg.data.test.test_mode = True  # 设置为测试模式（不使用数据增强）
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        # 如果每GPU样本数>1，需要替换图像张量处理方式
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        # 多个数据集情况（串联数据集）
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # ========== 分布式环境初始化 ==========
    # init distributed env first, since logger depends on the dist info.
    distributed = False  # 当前使用单GPU测试

    # ========== 随机种子设置 ==========
    # set random seeds - 设置随机种子以确保可复现性
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # ========== 数据加载器构建 ==========
    # build the dataloader - 构建测试数据集
    dataset = build_dataset(cfg.data.test)
    # 构建数据加载器
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,  # 每GPU的样本数
        workers_per_gpu=cfg.data.workers_per_gpu,  # 每GPU的数据加载进程数
        dist=distributed,  # 是否使用分布式
        shuffle=False,  # 测试时不打乱数据
    )

    # ========== 模型构建和加载 ==========
    # build the model and load checkpoint
    # 禁用训练配置（仅使用测试配置）
    cfg.model.train_cfg = None
    # 构建模型
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # ========== 混合精度处理 ==========
    # 如果启用fp16（半精度）混合精度训练
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)  # 包装模型以支持半精度推理
    # ========== 加载检查点 ==========
    # 从保存的检查点文件加载模型权重
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # ========== 卷积和批量归一化融合（可选） ==========
    # 融合Conv和BN层以加速推理
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # ========== 类别信息恢复 ==========
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # 旧版本检查点可能没有保存类别信息，需要从数据集恢复
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    # ========== 模型包装和测试 ==========
    if not distributed:
        # 单GPU测试：使用MMDataParallel包装
        model = MMDataParallel(model, device_ids=[0])
        # 执行单GPU测试
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        # 多GPU测试：使用MMDistributedDataParallel包装
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        # 执行多GPU测试
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    # ========== 结果处理（仅主进程执行） ==========
    # 获取当前进程的排名（多GPU时需要）
    rank, _ = get_dist_info()
    # 仅主进程（rank=0）处理和保存结果
    if rank == 0:
        # ========== 保存结果 ==========
        if args.out:
            print(f"\nwriting results to {args.out}")
            # 将预测结果保存为pickle文件
            mmcv.dump(outputs, args.out)
        # ========== 结果处理 ==========
        # 获取评估选项
        kwargs = {} if args.eval_options is None else args.eval_options
        # ========== 格式化结果（用于提交到测试服务器） ==========
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        # ========== 评估结果 ==========
        if args.eval:
            # 获取配置中的评估参数并进行深拷贝
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            # 移除不需要的EvalHook参数
            for key in [
                "interval",  # 评估间隔
                "tmpdir",  # 临时目录
                "start",  # 起始epoch
                "gpu_collect",  # GPU收集
                "save_best",  # 保存最优
                "rule",  # 比较规则
            ]:
                eval_kwargs.pop(key, None)
            # 添加命令行指定的评估指标
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            # 执行数据集评估并打印结果
            print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == "__main__":
    main()
