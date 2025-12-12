import argparse
import os
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

import torch
from datasets import Dataset, concatenate_datasets
from modelscope.msdatasets import MsDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def format_dataset_for_sft(ds, text_column: str = "text", max_length: int = 2048, tokenizer=None):
    """将数据集格式化为SFT训练所需的格式（使用Qwen3 chat template）

    Args:
        ds: 数据集对象
        text_column: 文本列名，如果不存在则尝试从对话格式构建
        max_length: 最大序列长度
        tokenizer: 分词器，用于应用chat template（Qwen3系列）
    """
    # 打印原始数据集信息
    print(f"原始数据集大小: {len(ds)}")
    print(f"原始数据集列名: {ds.column_names}")
    if len(ds) > 0:
        print(f"第一个样本的键: {list(ds[0].keys())}")
        print(f"第一个样本示例（前200字符）: {str(ds[0])[:200]}")

    # 检查是否可以使用chat template
    use_chat_template = tokenizer is not None and hasattr(tokenizer, "apply_chat_template")
    if use_chat_template:
        print("使用Qwen3 chat template格式化数据")
    else:
        print("警告: 未提供tokenizer或tokenizer不支持chat template，将使用简单格式")

    def format_conversation(example):
        """格式化对话数据为文本格式（使用Qwen3 chat template）"""
        # 如果已经有text字段，检查是否已经是chat template格式
        if text_column in example and example[text_column]:
            text_val = example[text_column]
            if text_val and str(text_val).strip():
                # 如果已经包含chat template标记，直接返回
                if "<|im_start|>" in str(text_val) or "<|im_end|>" in str(text_val):
                    return {"text": str(text_val)}
                # 否则尝试用chat template重新格式化（如果有conversations信息）
                # 这里先返回原文本，后续可以改进

        # 尝试从对话格式构建（优先使用chat template）
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
                # 构建messages列表用于chat template
                messages = []
                for msg in conversations:
                    if isinstance(msg, dict):
                        role = msg.get("role", msg.get("from", ""))
                        content = msg.get("content", msg.get("value", ""))
                        if role and content:
                            # 标准化角色名称
                            if role in ["user", "human"]:
                                messages.append({"role": "user", "content": str(content)})
                            elif role in ["assistant", "gpt", "bot"]:
                                messages.append({"role": "assistant", "content": str(content)})
                            else:
                                messages.append({"role": role, "content": str(content)})
                
                # 使用chat template格式化
                if messages and use_chat_template:
                    try:
                        formatted_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False  # 训练时不需要generation prompt
                        )
                        return {"text": formatted_text}
                    except Exception as e:
                        print(f"警告: 使用chat template失败: {e}，回退到简单格式")
                        # 回退到简单格式
                        text_parts = []
                        for msg in messages:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "user":
                                text_parts.append(f"用户: {content}")
                            elif role == "assistant":
                                text_parts.append(f"助手: {content}")
                            else:
                                text_parts.append(f"{role}: {content}")
                        if text_parts:
                            return {"text": "\n".join(text_parts)}
                elif messages:
                    # 没有chat template，使用简单格式
                    text_parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user":
                            text_parts.append(f"用户: {content}")
                        elif role == "assistant":
                            text_parts.append(f"助手: {content}")
                        else:
                            text_parts.append(f"{role}: {content}")
                    if text_parts:
                        return {"text": "\n".join(text_parts)}

        # 尝试从prompt和response构建
        if "prompt" in example and "response" in example:
            prompt = example.get("prompt", "")
            response = example.get("response", "")
            if prompt and response:
                if use_chat_template:
                    try:
                        messages = [
                            {"role": "user", "content": str(prompt)},
                            {"role": "assistant", "content": str(response)}
                        ]
                        formatted_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        return {"text": formatted_text}
                    except Exception as e:
                        print(f"警告: 使用chat template失败: {e}，回退到简单格式")
                return {"text": f"用户: {prompt}\n助手: {response}"}

        # 尝试从input和output构建
        if "input" in example and "output" in example:
            input_text = example.get("input", "")
            output_text = example.get("output", "")
            if input_text and output_text:
                if use_chat_template:
                    try:
                        messages = [
                            {"role": "user", "content": str(input_text)},
                            {"role": "assistant", "content": str(output_text)}
                        ]
                        formatted_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        return {"text": formatted_text}
                    except Exception as e:
                        print(f"警告: 使用chat template失败: {e}，回退到简单格式")
                return {"text": f"用户: {input_text}\n助手: {output_text}"}

        # 尝试从instruction和output构建
        if "instruction" in example and "output" in example:
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            if instruction and output:
                if use_chat_template:
                    try:
                        messages = [
                            {"role": "user", "content": str(instruction)},
                            {"role": "assistant", "content": str(output)}
                        ]
                        formatted_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        return {"text": formatted_text}
                    except Exception as e:
                        print(f"警告: 使用chat template失败: {e}，回退到简单格式")
                return {"text": f"用户: {instruction}\n助手: {output}"}

        # 如果都没有，尝试将所有非空字段拼接
        text_parts = []
        for key, value in example.items():
            if value and isinstance(value, (str, int, float)):
                text_parts.append(f"{key}: {value}")
        if text_parts:
            return {"text": "\n".join(text_parts)}

        # 如果都没有，返回空字符串
        return {"text": ""}

    # 应用格式化函数
    formatted_ds = ds.map(format_conversation, remove_columns=ds.column_names)
    print(f"格式化后数据集大小（过滤前）: {len(formatted_ds)}")

    # 检查格式化后的数据
    if len(formatted_ds) > 0:
        non_empty_count = sum(1 for i in range(min(10, len(formatted_ds))) if formatted_ds[i]["text"].strip())
        print(f"前10个样本中非空文本数量: {non_empty_count}")
        if non_empty_count > 0:
            print(f"第一个非空样本示例: {formatted_ds[0]['text'][:200]}")

    # 过滤空文本
    formatted_ds = formatted_ds.filter(lambda x: len(x["text"].strip()) > 0)
    print(f"过滤后数据集大小: {len(formatted_ds)}")

    return formatted_ds


def load_dataset_for_sft(
        dataset_name_or_path: Optional[str] = None,
        subset_name: Optional[Union[str, List[str]]] = None,
        split: str = "train",
        text_column: str = "text",
        max_length: int = 2048,
        tokenizer=None,
) -> Dataset:
    """加载并格式化数据集用于SFT训练"""
    if dataset_name_or_path:
        # 如果是本地路径
        if os.path.exists(dataset_name_or_path):
            from datasets import load_dataset
            ds = load_dataset("json", data_files=dataset_name_or_path, split=split)
        else:
            # 尝试从ModelScope加载
            try:
                if subset_name:
                    if isinstance(subset_name, list):
                        # 加载多个子集并合并
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
                # 如果已经是HuggingFace Dataset，直接使用
            except Exception as e:
                print(f"从ModelScope加载失败，尝试从HuggingFace加载: {e}")
                from datasets import load_dataset
                ds = load_dataset(dataset_name_or_path, split=split)
    else:
        # 使用默认数据集，加载所有可用的子集
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

        # 如果指定了subset_name，使用指定的；否则使用所有默认子集
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

    # 检查数据集是否为空
    if len(ds) == 0:
        print("警告: 加载的数据集为空！")
        print("请检查数据集路径或名称是否正确。")
        return ds

    # 格式化数据集（使用Qwen3 chat template）
    formatted_ds = format_dataset_for_sft(ds, text_column=text_column, max_length=max_length, tokenizer=tokenizer)

    # 清理原始数据集以释放内存
    del ds
    import gc
    gc.collect()

    if len(formatted_ds) == 0:
        print("警告: 格式化后的数据集为空！")
        print("可能的原因:")
        print("1. 数据集格式不符合预期")
        print("2. 所有数据都被过滤掉了")
        print("3. 数据格式解析失败")
        print(f"请检查原始数据集的列名和格式: {ds.column_names}")
        if len(ds) > 0:
            print(f"原始数据示例: {ds[0]}")

    return formatted_ds


@dataclass
class DataCollatorForCausalLM:
    """用于通用文本全量微调的 data collator。

    功能：
    - 使用 tokenizer 对样本中的 `text` 字段做截断和 padding
    - 构造 labels，并将 padding 位置的标签置为 -100（忽略这些位置的 loss）
    - 对非 padding 的 token 全量计算 loss，适用于通用语言建模目标
    """

    tokenizer: AutoTokenizer
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 从样本中取出文本字段
        texts = [f["text"] for f in features]

        # 统一做 tokenize / 截断 / padding
        batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # labels 初始与 input_ids 一致
        labels = batch["input_ids"].clone()

        # 将 padding 位置的 label 置为 -100，避免对 padding 计算 loss
        if "attention_mask" in batch:
            pad_mask = batch["attention_mask"] == 0
            labels[pad_mask] = -100

        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT训练 (TRL)")
    parser.add_argument("--model_name", type=str, default="/data/Qwen3-32B",
                        help="模型路径或名称")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集名称或路径，如果为None则使用默认数据集")
    parser.add_argument("--subset_name", default=None, nargs='+',
                        help="数据集子集名称（用于ModelScope），可以指定多个，用空格分隔。如果不指定，则使用所有默认子集")
    parser.add_argument("--text_column", type=str, default="text",
                        help="文本列名")
    parser.add_argument("--output_dir", type=str, default="./outputs-sft",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--bf16", action="store_true",
                        help="使用bfloat16精度")
    parser.add_argument("--fp16", action="store_true",
                        help="使用float16精度")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="数据加载器工作进程数")
    parser.add_argument("--dataloader_pin_memory", action="store_true",
                        help="启用数据加载器pin memory")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="预热步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存检查点的步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--eval_strategy", type=str, default="no",
                        choices=["no", "steps", "epoch"],
                        help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估步数（当eval_strategy=steps时）")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="推送到HuggingFace Hub")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="最大训练步数，-1表示使用num_train_epochs")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="启用梯度检查点以节省内存")
    parser.add_argument("--packing", action="store_true",
                        help="启用序列打包以提高效率")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="保存的检查点数量限制")
    parser.add_argument("--max_memory_MB", type=int, default=None,
                        help="每个GPU的最大显存限制（MB），用于模型加载时的内存管理")
    parser.add_argument("--use_8bit_optimizer", action="store_true",
                        help="使用8-bit优化器（bitsandbytes），可大幅减少优化器状态显存（节省约50-75%优化器显存）")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="优化器类型，如果使用8-bit优化器，应设置为adamw_8bit或paged_adamw_8bit")

    args = parser.parse_args()
    set_seed(args.seed)

    # 加载模型和分词器
    print(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 检测分布式训练环境（torchrun会自动设置环境变量和初始化进程组）
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        # torchrun 已经初始化了进程组，这里只需要设置设备
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"分布式训练: 进程 {rank}/{world_size} 在设备 {device} 上运行")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"单GPU训练模式，使用设备: {device}")

    # 优化模型加载以减少显存占用
    # 注意：当使用Trainer的FP16/BF16选项时，不应该在加载时指定torch_dtype
    # Trainer会自动处理混合精度训练，包括模型的数据类型转换
    # 如果在加载时指定了torch_dtype，会导致梯度缩放冲突错误
    # 使用low_cpu_mem_usage可以减少CPU内存占用
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,  # 减少CPU内存占用
    }

    # 如果指定了显存限制，使用max_memory参数
    # 注意：在DDP模式下不能使用device_map，因为DDP会自己管理设备分配
    if args.max_memory_MB is not None and not ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        # 只在单GPU训练时使用device_map和max_memory
        max_memory = {0: f"{args.max_memory_MB}MB"}
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"  # 使用device_map自动分配
    elif args.max_memory_MB is not None:
        print(f"警告: 在分布式训练模式下，max_memory参数将被忽略（DDP模式不支持device_map）")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )

    # 清理显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # 将模型移动到对应设备
    # 如果使用了device_map="auto"，模型已经自动分配到设备，不需要再次移动
    if "device_map" not in model_kwargs:
        model = model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 加载和格式化数据集（使用Qwen3 chat template）
    print("加载数据集...")
    train_ds = load_dataset_for_sft(
        dataset_name_or_path=args.dataset,
        subset_name=args.subset_name,
        split="train",
        text_column=args.text_column,
        max_length=args.max_seq_length,
        tokenizer=tokenizer,  # 传递tokenizer以使用chat template
    )
    print(f"训练集大小: {len(train_ds)}")

    # 配置训练参数
    sft_cfg_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "eval_strategy": args.eval_strategy,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": False,
        "fp16": args.fp16 and not args.bf16,
        "bf16": args.bf16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "report_to": ["none"],
        "push_to_hub": args.push_to_hub,
        "gradient_checkpointing": args.gradient_checkpointing,
        "packing": args.packing,
        "remove_unused_columns": False,
        "dataset_text_field": args.text_column,
    }

    # 如果使用8-bit优化器，设置优化器类型
    if args.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            print("bitsandbytes已安装，可以使用8-bit优化器")
        except ImportError:
            print("警告: 未安装bitsandbytes，8-bit优化器将无法使用")
            print("请运行: pip install bitsandbytes")
            print("将回退到标准优化器")
            args.use_8bit_optimizer = False

        if args.use_8bit_optimizer:
            if args.optim == "adamw_torch":
                # 默认使用paged_adamw_8bit，这是最节省显存的8-bit优化器
                sft_cfg_kwargs["optim"] = "paged_adamw_8bit"
            else:
                sft_cfg_kwargs["optim"] = args.optim
            print("使用8-bit优化器以节省显存（可节省约50-75%优化器状态显存）")

    # 只在 max_steps > 0 时添加该参数
    if args.max_steps > 0:
        sft_cfg_kwargs["max_steps"] = args.max_steps

    # 只在 eval_strategy 为 "steps" 时添加 eval_steps
    if args.eval_strategy == "steps":
        sft_cfg_kwargs["eval_steps"] = args.eval_steps

    sft_cfg = SFTConfig(**sft_cfg_kwargs)

    # 为通用文本全量微调创建 data collator：
    # - 基于 `text` 字段做 tokenize / 截断 / padding
    # - 仅对非 padding token 计算 loss（padding 位置 label=-100）
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )

    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 开始训练前清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(
            f"训练前显存状态: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB / {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存模型（只在主进程保存）
    if rank == 0:
        print(f"保存模型到: {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        print(f"进程 {rank} 跳过模型保存（由主进程负责）")

    print("训练完成！")


if __name__ == "__main__":
    main()
