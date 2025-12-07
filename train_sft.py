import argparse
import os
from typing import Optional, Union, List

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset, concatenate_datasets
from modelscope.msdatasets import MsDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)
from trl import PPOConfig, PPOTrainer





def ensure_model_config_access(model):
    """确保模型能够正确访问 config 属性，即使被 DDP 包装
    
    当模型被 DistributedDataParallel 包装时，直接访问 model.config 会失败。
    按照参考方案，直接解包为原始模型（ddp_model.module）。
    """
    if isinstance(model, nn.parallel.DistributedDataParallel):
        # 直接返回原始模型，而不是 DDP 包装
        return model.module
    return model


def build_reward_model(
        reward_model_name: str,
        torch_dtype: Optional[torch.dtype] = torch.float16,
) -> AutoModelForSequenceClassification:
    """Create a reward model using a sequence classification model.

    Returns a nn.Module that outputs reward scores.
    """
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        torch_dtype=torch_dtype,
        # device_map="auto"
    )
    return rm_model


def get_default_dataset(num_examples: int = 50) -> Dataset:
    prompts = [
        "Summarize the benefits of unit testing.",
        "Explain the difference between synchronous and asynchronous programming.",
        "List three tips for writing readable code.",
        "What is overfitting and how to prevent it?",
        "Describe a use-case for message queues.",
    ]
    prompts = (prompts * ((num_examples + len(prompts) - 1) // len(prompts)))[:num_examples]
    return Dataset.from_dict({"prompt": prompts})


def extract_prompt_from_conversation(example):
    """从对话数据中提取prompt（用户输入）"""
    # 如果已经有prompt字段，直接返回
    if "prompt" in example and example["prompt"]:
        return {"prompt": str(example["prompt"])}

    # 尝试从对话格式中提取用户输入
    if "conversations" in example or "messages" in example:
        conversations = example.get("conversations") or example.get("messages", [])
        if isinstance(conversations, str):
            # 如果是字符串，尝试解析JSON
            import json
            try:
                conversations = json.loads(conversations)
            except:
                pass
        if isinstance(conversations, list) and len(conversations) > 0:
            # 取第一条用户消息作为prompt
            for msg in conversations:
                if isinstance(msg, dict):
                    role = msg.get("role", msg.get("from", ""))
                    content = msg.get("content", msg.get("value", ""))
                    if (role == "user" or role == "human") and content:
                        return {"prompt": str(content)}

    # 尝试从其他字段提取
    for key in ["input", "instruction", "question"]:
        if key in example and example[key]:
            return {"prompt": str(example[key])}

    # 如果都没有，返回空字符串
    return {"prompt": ""}


def load_prompts_dataset(
        dataset_name_or_path: Optional[str] = None,
        subset_name: Optional[Union[str, List[str]]] = None,
        split: str = "train",
        prompt_column: str = "prompt"
) -> Dataset:
    """加载prompt数据集，支持本地文件、HuggingFace和ModelScope"""
    if not dataset_name_or_path:
        # 使用默认数据集 AI-ModelScope/COIG-CQIA
        default_subsets = [
            'chinese_traditional',
            'coig_pc',
            'exam',
            'finance',
            'douban',
            'human_value',
            'logi_qa',
            'ruozhiba',
            'segmentfault',
            'wiki',
            'wikihow',
            'xhs',
            'zhihu'
        ]

        if subset_name:
            if isinstance(subset_name, str):
                subset_list = [subset_name]
            else:
                subset_list = subset_name
        else:
            subset_list = default_subsets

        print(f"使用默认数据集: AI-ModelScope/COIG-CQIA")
        print(f"加载子集: {subset_list}")

        datasets = []
        for subset in subset_list:
            try:
                print(f"正在加载子集: {subset}...")
                subset_ds = MsDataset.load('AI-ModelScope/COIG-CQIA', subset_name=subset, split='train')
                # 转换为HuggingFace Dataset格式（如果还不是）
                if hasattr(subset_ds, 'to_hf_dataset'):
                    subset_ds = subset_ds.to_hf_dataset()
                print(f"子集 {subset} 加载完成，大小: {len(subset_ds)}")
                datasets.append(subset_ds)
            except Exception as e:
                print(f"警告: 加载子集 {subset} 失败: {e}")
                continue

        if not datasets:
            raise ValueError("所有子集加载失败，请检查数据集名称和网络连接")

        print(f"合并 {len(datasets)} 个子集...")
        ds = concatenate_datasets(datasets)
        print(f"合并后数据集总大小: {len(ds)}")

        # 从对话数据中提取prompt
        print("从对话数据中提取prompt...")
        ds = ds.map(extract_prompt_from_conversation, remove_columns=ds.column_names)
        # 过滤空prompt
        ds = ds.filter(lambda x: len(x["prompt"].strip()) > 0)
        print(f"提取prompt后数据集大小: {len(ds)}")

        return ds.select_columns(["prompt"])

    # 如果是本地路径
    if os.path.exists(dataset_name_or_path):
        ds = load_dataset("json", data_files=dataset_name_or_path, split=split)
    else:
        # 尝试从ModelScope加载
        try:
            if subset_name:
                if isinstance(subset_name, list):
                    datasets = []
                    for subset in subset_name:
                        print(f"加载子集: {subset}")
                        subset_ds = MsDataset.load(dataset_name_or_path, subset_name=subset, split=split)
                        if hasattr(subset_ds, 'to_hf_dataset'):
                            subset_ds = subset_ds.to_hf_dataset()
                        datasets.append(subset_ds)
                    print(f"合并 {len(datasets)} 个子集...")
                    ds = concatenate_datasets(datasets)
                else:
                    ds = MsDataset.load(dataset_name_or_path, subset_name=subset_name, split=split)
            else:
                ds = MsDataset.load(dataset_name_or_path, split=split)
            # 转换为HuggingFace Dataset格式（如果还不是）
            if hasattr(ds, 'to_hf_dataset'):
                ds = ds.to_hf_dataset()
            # 从对话数据中提取prompt
            ds = ds.map(extract_prompt_from_conversation, remove_columns=ds.column_names)
            ds = ds.filter(lambda x: len(x["prompt"].strip()) > 0)
        except Exception as e:
            print(f"从ModelScope加载失败，尝试从HuggingFace加载: {e}")
            ds = load_dataset(dataset_name_or_path, split=split)

    # ensure prompt column
    if prompt_column != "prompt" and prompt_column in ds.column_names:
        ds = ds.rename_column(prompt_column, "prompt")

    return ds.select_columns(["prompt"]) if "prompt" in ds.column_names else get_default_dataset()


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training with combined rewards (TRL)")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--reward_model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集名称或路径，如果为None则使用默认数据集 AI-ModelScope/COIG-CQIA")
    parser.add_argument("--subset_name", default=None, nargs='+',
                        help="数据集子集名称（用于ModelScope），可以指定多个，用空格分隔。如果不指定，则使用所有默认子集")
    parser.add_argument("--prompt_column", type=str, default="prompt",
                        help="prompt列名（用于非ModelScope数据集）")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="最大序列长度，用于对prompt进行截断")
    parser.add_argument("--output_dir", type=str, default="./outputs-ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=2, help="Samples per prompt for PPO")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Enable pin memory for dataloader")
    parser.add_argument("--vf_coef", type=float, default=0.1, help="Value function loss coefficient")
    parser.add_argument("--cliprange", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Override total training steps when dataset has no length")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="每多少步保存一次模型")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="每多少步记录一次日志")
    parser.add_argument(
        "--use_8bit_optimizer",
        action="store_true",
        help="使用8-bit优化器（bitsandbytes），可大幅减少优化器状态显存（节省约50-75%优化器显存）",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # Load policy model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine torch dtype for model
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #   torch_dtype=torch_dtype,
        # device_map="auto",
    )

    # Build reward model
    reward_model = build_reward_model(args.reward_model_name)
    # 设置 reward_model 为 eval 模式，不参与训练
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    # Data
    train_ds = load_prompts_dataset(
        dataset_name_or_path=args.dataset,
        subset_name=args.subset_name,
        split="train",
        prompt_column=args.prompt_column
    )
    print(f"原始训练集大小: {len(train_ds)}")

    # 参考 train_sft.py，对数据集做一次标准化/编码：
    # 将只包含 'prompt' 文本的样本，转换为包含 input_ids / attention_mask 的编码，
    # 这样后续的 DataLoader / Trainer 在取 batch 时拿到的就是合法的编码，而不是只含 'prompt' 的字典，
    # 避免 "provided ['prompt']" 这类错误。
    def tokenize_prompt_batch(examples):
        texts = [str(t) for t in examples["prompt"]]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        return encodings

    train_ds = train_ds.map(
        tokenize_prompt_batch,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    print(f"编码后训练集大小: {len(train_ds)}，列名: {train_ds.column_names}")

    # 构建 PPOConfig 参数字典，方便后续根据开关动态调整
    ppo_cfg_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # 关闭 TRL 内部的 gradient checkpointing，避免对 PolicyAndValueWrapper
        # 调用不存在的 gradient_checkpointing_disable 导致 AttributeError
        "gradient_checkpointing": False,
        "vf_coef": args.vf_coef,
        "cliprange": args.cliprange,
        "cliprange_value": args.cliprange_value,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "eval_strategy": "no",
        "remove_unused_columns": False,
        # 关闭 TRL/accelerate 的 FP16 训练，避免 GradScaler 抛出
        # "Attempting to unscale FP16 gradients." 一类错误。
        # 如需混合精度，推荐在支持的 GPU 上使用 bfloat16（通过 --bf16 开启）。
        "fp16": False,
        "bf16": args.bf16,
        # "mini_batch_size": args.num_generations,
        "report_to": ["none"],
        "push_to_hub": args.push_to_hub,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
    }

    # 如果使用8-bit优化器，则通过 optim 字段交给 TRL/transformers 选择 8bit 优化器
    if args.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb  # noqa: F401
            print("bitsandbytes已安装，可以在PPO中使用8-bit优化器")
            # 默认使用 paged_adamw_8bit，与 SFT 部分保持一致
            ppo_cfg_kwargs["optim"] = "paged_adamw_8bit"
            print("PPO 将使用 paged_adamw_8bit 优化器（可节省约50-75%优化器状态显存）")
        except ImportError:
            print("警告: 未安装bitsandbytes，8-bit优化器将无法使用")
            print("请运行: pip install bitsandbytes")
            print("将回退到标准优化器（adamw_torch）")

    ppo_cfg = PPOConfig(**ppo_cfg_kwargs)

    # 一些 TRL 版本在构建内部 DataLoader / Sampler 时，如果 eval_dataset 为 None，
    # 会出现 sampler.data_source 为 None，进而在 len(self.data_source) 时报
    # "TypeError: object of type 'NoneType' has no len()"。
    # 这里显式把 eval_dataset 设成一个非空子集，既能提供用于 generate_completions
    # 的 prompt，又能避免上述报错。
    if len(train_ds) == 0:
        raise ValueError("训练数据集为空，请检查数据加载配置（dataset / subset_name 等）。")

    # 评估/生成用的数据集：从训练集里取前若干条即可
    eval_sample_size = min(len(train_ds), 128)
    eval_ds = train_ds.select(range(eval_sample_size))

    # 确保模型能够正确访问 config 属性（处理 DDP 包装的情况）
    model = ensure_model_config_access(model)
    reward_model = ensure_model_config_access(reward_model)

    trainer = PPOTrainer(
        model=model,
        ref_model=None,  # Will use the same model
        processing_class=tokenizer,
        args=ppo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        value_model=reward_model,
        reward_model=reward_model,
    )
    
    # PPOTrainer 内部可能会通过 accelerator.prepare() 包装模型为 DDP
    # 按照参考方案，如果模型被包装成 DDP，则解包为原始模型（ddp_model.module）
    if hasattr(trainer, 'model') and isinstance(trainer.model, nn.parallel.DistributedDataParallel):
        trainer.model = trainer.model.module
    if hasattr(trainer, 'ref_model') and trainer.ref_model is not None:
        if isinstance(trainer.ref_model, nn.parallel.DistributedDataParallel):
            trainer.ref_model = trainer.ref_model.module
    if hasattr(trainer, 'value_model') and trainer.value_model is not None:
        if isinstance(trainer.value_model, nn.parallel.DistributedDataParallel):
            trainer.value_model = trainer.value_model.module
    if hasattr(trainer, 'reward_model') and trainer.reward_model is not None:
        if isinstance(trainer.reward_model, nn.parallel.DistributedDataParallel):
            trainer.reward_model = trainer.reward_model.module

    trainer.train()
    
    # 检测分布式训练环境并处理模型保存
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # 导入分布式模块
        import torch.distributed as dist
        
        # 确保所有进程都完成训练并到达保存点
        if dist.is_initialized():
            print(f"进程 {rank}/{world_size} 训练完成，等待所有进程同步...")
            dist.barrier()
            print(f"进程 {rank}/{world_size} 已到达保存同步点")
        
        # 只在主进程保存模型，避免多进程同时写入导致冲突和NCCL超时
        if rank == 0:
            print(f"主进程保存模型到: {args.output_dir}")
            try:
                trainer.save_model(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print("模型保存完成")
            except Exception as e:
                print(f"保存模型时出错: {e}")
                raise
        else:
            print(f"进程 {rank} 跳过模型保存（由主进程负责）")
        
        # 保存完成后再次同步，确保所有进程都知道保存已完成
        if dist.is_initialized():
            dist.barrier()
            print(f"进程 {rank}/{world_size} 保存流程完成")
    else:
        # 单GPU训练模式
        print(f"保存模型到: {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("模型保存完成")



if __name__ == "__main__":
    main()
