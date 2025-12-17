import argparse
import os
from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from modelscope.msdatasets import MsDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    set_seed,
    TrainerCallback,
)

# 延迟导入 trl 以避免配置冲突
# 如果 aimv2 配置已被注册（例如由 vLLM），在导入前处理冲突
def _import_trl():
    """安全导入 trl，处理配置冲突"""
    # 在导入 trl 之前，先处理可能的配置冲突
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    
    # 临时修改 register 方法以允许已存在的配置
    original_register = CONFIG_MAPPING.register
    
    def patched_register(key, value, exist_ok=False):
        """修补的注册函数，对 aimv2 使用 exist_ok=True"""
        if key == "aimv2":
            # 对于 aimv2，检查是否已存在
            # CONFIG_MAPPING 可能有不同的内部结构，尝试多种方式检查
            config_exists = False
            try:
                # 方法1: 检查 _extra_content
                if hasattr(CONFIG_MAPPING, '_extra_content'):
                    config_exists = key in CONFIG_MAPPING._extra_content
                # 方法2: 尝试直接访问（如果已存在会返回配置类）
                if not config_exists:
                    try:
                        _ = CONFIG_MAPPING[key]
                        config_exists = True
                    except (KeyError, AttributeError):
                        pass
            except Exception:
                pass
            
            # 如果配置已存在，跳过注册
            if config_exists:
                return
            
            # 尝试使用 exist_ok=True 注册
            try:
                return original_register(key, value, exist_ok=True)
            except (TypeError, ValueError) as e:
                # 如果 exist_ok 参数不支持或仍然失败，检查是否是"已存在"错误
                if "already used" in str(e) or "already exists" in str(e).lower():
                    # 配置已存在，这是预期的，直接返回
                    return
                # 其他错误，重新抛出
                raise
        else:
            return original_register(key, value, exist_ok=exist_ok)
    
    # 应用补丁
    CONFIG_MAPPING.register = patched_register
    
    try:
        from trl import GRPOConfig, GRPOTrainer
        return GRPOConfig, GRPOTrainer
    except ValueError as e:
        if "'aimv2'" in str(e) or "already used by a Transformers config" in str(e):
            import warnings
            import sys
            
            warnings.warn(
                f"检测到配置冲突 (aimv2): {e}。"
                "这通常由 vLLM 或其他库已注册该配置导致。"
                "尝试清除模块缓存并重新导入..."
            )
            
            # 清除 trl 相关模块缓存
            modules_to_clear = [k for k in list(sys.modules.keys()) if k.startswith('trl')]
            for mod in modules_to_clear:
                del sys.modules[mod]
            
            # 再次尝试导入
            try:
                from trl import GRPOConfig, GRPOTrainer
                return GRPOConfig, GRPOTrainer
            except ValueError:
                # 如果仍然失败，恢复原始函数并抛出错误
                CONFIG_MAPPING.register = original_register
                raise RuntimeError(
                    f"无法解决配置冲突: {e}。"
                    "请检查是否安装了冲突的库（如 vLLM），"
                    "或尝试更新 transformers 和 trl 库的版本。"
                ) from e
        else:
            # 恢复原始函数
            CONFIG_MAPPING.register = original_register
            raise
    finally:
        # 确保恢复原始函数（即使成功导入）
        CONFIG_MAPPING.register = original_register

# 导入 trl 组件
GRPOConfig, GRPOTrainer = _import_trl()


class DistributedSaveCallback(TrainerCallback):
    """自定义回调，确保在分布式训练中保存模型时正确同步，避免NCCL超时"""
    
    def _check_distributed(self):
        """检查分布式环境是否已初始化"""
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return True, dist, int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        return False, None, 0, 1
    
    def on_save(self, args, state, control, **kwargs):
        """在保存检查点前同步所有进程"""
        is_dist, dist, rank, world_size = self._check_distributed()
        if is_dist:
            # 确保所有进程都到达保存点，避免部分进程提前尝试保存导致NCCL超时
            dist.barrier()
            if rank == 0:
                print(f"主进程 (rank {rank}/{world_size}) 正在保存检查点...")
            else:
                print(f"进程 {rank}/{world_size} 已到达保存同步点，等待主进程保存...")
    
    def on_save_end(self, args, state, control, **kwargs):
        """保存完成后同步所有进程"""
        is_dist, dist, rank, world_size = self._check_distributed()
        if is_dist:
            # 确保所有进程都知道保存已完成，避免后续操作不同步
            dist.barrier()
            if rank == 0:
                print(f"检查点保存完成 (rank {rank}/{world_size})")
            else:
                print(f"进程 {rank}/{world_size} 已收到保存完成信号")




def build_reward_model_fn(
        reward_model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        torch_dtype: Optional[torch.dtype] = torch.float16,
) -> Callable[[List[str], List[str]], List[float]]:
    """Create a reward function using a sequence classification model.

    Returns a function that outputs a scalar reward per completion.
    """
    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, use_fast=True, trust_remote_code=True)

    # ensure padding token exists for batched inference
    if rm_tokenizer.pad_token is None:
        candidate = rm_tokenizer.eos_token or rm_tokenizer.sep_token or rm_tokenizer.cls_token or rm_tokenizer.unk_token
        if candidate is not None:
            rm_tokenizer.pad_token = candidate
        else:
            rm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    rm_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, torch_dtype=torch_dtype,
                                                                  device_map="auto", trust_remote_code=True)
    if getattr(rm_model.config, "pad_token_id", None) is None and rm_tokenizer.pad_token_id is not None:
        rm_model.config.pad_token_id = rm_tokenizer.pad_token_id

    # 不使用 pipeline，直接使用模型和 tokenizer 以避免类型转换问题
    rm_model.eval()
    # 如果使用 device_map="auto"，模型可能分布在多个设备上，需要从模型参数获取设备
    if hasattr(rm_model, "hf_device_map") and rm_model.hf_device_map:
        # 模型使用了 device_map，从第一个参数获取设备
        first_param = next(rm_model.parameters())
        device = str(first_param.device)
    else:
        # 否则使用指定的 device 或默认设备
        device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(rm_model, "to"):
            rm_model = rm_model.to(device)

    def reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        del prompts  # unused here
        batch_size = kwargs.get("batch_size", 2)
        scores: List[float] = []
        
        # 分批处理，确保 input_ids 是正确的整数类型
        for i in range(0, len(completions), batch_size):
            batch_completions = completions[i:i + batch_size]
            
            # 过滤掉空字符串，记录非空 completion 在批次中的位置
            valid_completions = []
            valid_positions = []  # 在批次中的位置索引
            for pos, comp in enumerate(batch_completions):
                if comp and comp.strip():  # 非空且非纯空白
                    valid_completions.append(comp)
                    valid_positions.append(pos)
            
            # 如果批次中所有 completion 都为空，给它们分配默认低分
            if not valid_completions:
                batch_scores = [-1.0] * len(batch_completions)  # 空 completion 给低分
                scores.extend(batch_scores)
                continue
            
            # 使用 tokenizer 编码，确保返回的是整数类型的 input_ids
            encoded = rm_tokenizer(
                valid_completions,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,  # 设置合理的最大长度
            )
            # 确保 input_ids 是 Long 类型（整数）
            # 使用 to(torch.long) 而不是 long()，并先检查类型
            input_ids = encoded["input_ids"]
            
            # 检查 input_ids 是否为空（防止 tokenizer 产生空序列）
            # 检查：1) 总元素数为0 2) batch_size为0 3) 序列长度为0
            if input_ids.numel() == 0 or len(input_ids.shape) < 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
                # 如果编码后为空，给这些 completion 分配默认低分
                valid_scores = [-1.0] * len(valid_completions)
            else:
                if input_ids.dtype != torch.long:
                    input_ids = input_ids.to(torch.long)
                input_ids = input_ids.to(device)
                
                attention_mask = encoded.get("attention_mask", None)
                if attention_mask is not None:
                    # 确保 attention_mask 也是正确的类型
                    if attention_mask.dtype != torch.long:
                        attention_mask = attention_mask.to(torch.long)
                    attention_mask = attention_mask.to(device)
                
                # 前向传播
                with torch.no_grad():
                    if attention_mask is not None:
                        outputs = rm_model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        outputs = rm_model(input_ids=input_ids)
                    
                    # 获取 logits
                    logits = outputs.logits
                    
                    # 处理 logits：如果是二分类，使用正类；否则使用最后一个类
                    if logits.shape[-1] == 1:
                        valid_scores = logits[:, 0].cpu().tolist()
                    else:
                        # prefer last class as "more positive"
                        valid_scores = logits[:, -1].cpu().tolist()
            
            # 将有效 completion 的分数映射回原始批次位置
            batch_scores = [-1.0] * len(batch_completions)  # 默认低分
            for pos, score in zip(valid_positions, valid_scores):
                batch_scores[pos] = score
            
            scores.extend(batch_scores)
        
        if not normalize:
            return scores
        # z-norm for stability (per-batch)
        t = torch.tensor(scores, dtype=torch.float32)
        std = float(t.std().clamp(min=1e-6))
        mean = float(t.mean())
        normed = ((t - mean) / std).tolist()
        return [float(x) for x in normed]

    return reward_fn




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


def extract_prompt_from_conversation(example, tokenizer=None):
    """从对话数据中提取prompt（用户输入），使用Qwen3 chat template格式化
    
    Args:
        example: 数据集样本
        tokenizer: 分词器，用于应用chat template（Qwen3系列）
    """
    # 检查是否可以使用chat template
    use_chat_template = tokenizer is not None and hasattr(tokenizer, "apply_chat_template")
    
    # 如果已经有prompt字段，使用chat template格式化
    if "prompt" in example and example["prompt"]:
        prompt_text = str(example["prompt"])
        # 如果已经包含chat template标记，直接返回
        if "<|im_start|>" in prompt_text or "<|im_end|>" in prompt_text:
            return {"prompt": prompt_text}
        # 否则使用chat template格式化（只包含用户消息）
        if use_chat_template:
            try:
                messages = [{"role": "user", "content": prompt_text}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return {"prompt": formatted_prompt}
            except Exception as e:
                print(f"警告: 使用chat template格式化prompt失败: {e}，使用原始prompt")
        return {"prompt": prompt_text}

    # 尝试从对话格式中提取用户输入
    if "conversations" in example or "messages" in example:
        conversations = example.get("conversations") or example.get("messages", [])
        if isinstance(conversations, str):
            # 如果是字符串，尝试解析JSON
            import json
            try:
                conversations = json.loads(conversations)
            except Exception:
                pass
        if isinstance(conversations, list) and len(conversations) > 0:
            # 取第一条用户消息作为prompt
            for msg in conversations:
                if isinstance(msg, dict):
                    role = msg.get("role", msg.get("from", ""))
                    content = msg.get("content", msg.get("value", ""))
                    if (role == "user" or role == "human") and content:
                        prompt_text = str(content)
                        # 使用chat template格式化（只包含用户消息）
                        if use_chat_template:
                            try:
                                messages = [{"role": "user", "content": prompt_text}]
                                formatted_prompt = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=False
                                )
                                return {"prompt": formatted_prompt}
                            except Exception as e:
                                print(f"警告: 使用chat template格式化prompt失败: {e}，使用原始prompt")
                        return {"prompt": prompt_text}

    # 尝试从其他字段提取
    for key in ["input", "instruction", "question"]:
        if key in example and example[key]:
            prompt_text = str(example[key])
            # 使用chat template格式化（只包含用户消息）
            if use_chat_template:
                try:
                    messages = [{"role": "user", "content": prompt_text}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    return {"prompt": formatted_prompt}
                except Exception as e:
                    print(f"警告: 使用chat template格式化prompt失败: {e}，使用原始prompt")
            return {"prompt": prompt_text}

    # 如果都没有，返回空字符串
    return {"prompt": ""}


def load_prompts_dataset(
    dataset_name_or_path: Optional[str] = None,
    subset_name: Optional[List[str]] = None,
    split: str = "train",
    prompt_column: str = "prompt",
    tokenizer=None,
) -> Dataset:
    """加载prompt数据集，支持本地文件、HuggingFace和ModelScope。

    和 PPO 版本对齐：默认使用 AI-ModelScope/COIG-CQIA 以及子集机制，
    并从对话数据中抽取 prompt 字段。
    """
    if not dataset_name_or_path:
        # 使用默认数据集 AI-ModelScope/COIG-CQIA
        default_subsets = [
            "chinese_traditional",
            "coig_pc",
            "exam",
            "finance",
            "douban",
            "human_value",
            "logi_qa",
            "ruozhiba",
            "segmentfault",
            "wiki",
            "wikihow",
            "xhs",
            "zhihu",
        ]

        if subset_name:
            subset_list = subset_name
        else:
            subset_list = default_subsets

        print("使用默认数据集: AI-ModelScope/COIG-CQIA")
        print(f"加载子集: {subset_list}")

        datasets = []
        for subset in subset_list:
            try:
                print(f"正在加载子集: {subset}...")
                subset_ds = MsDataset.load("AI-ModelScope/COIG-CQIA", subset_name=subset, split="train")
                # 转换为HuggingFace Dataset格式（如果还不是）
                if hasattr(subset_ds, "to_hf_dataset"):
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

        # 从对话数据中提取prompt（使用Qwen3 chat template）
        print("从对话数据中提取prompt（使用Qwen3 chat template）...")
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            print("使用Qwen3 chat template格式化prompt")
        ds = ds.map(lambda x: extract_prompt_from_conversation(x, tokenizer=tokenizer), remove_columns=ds.column_names)
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
                datasets = []
                for subset in subset_name:
                    print(f"加载子集: {subset}")
                    subset_ds = MsDataset.load(dataset_name_or_path, subset_name=subset, split=split)
                    if hasattr(subset_ds, "to_hf_dataset"):
                        subset_ds = subset_ds.to_hf_dataset()
                    datasets.append(subset_ds)
                print(f"合并 {len(datasets)} 个子集...")
                ds = concatenate_datasets(datasets)
            else:
                ds = MsDataset.load(dataset_name_or_path, split=split)
            # 转换为HuggingFace Dataset格式（如果还不是）
            if hasattr(ds, "to_hf_dataset"):
                ds = ds.to_hf_dataset()
            # 从对话数据中提取prompt（使用Qwen3 chat template）
            ds = ds.map(lambda x: extract_prompt_from_conversation(x, tokenizer=tokenizer), remove_columns=ds.column_names)
            ds = ds.filter(lambda x: len(x["prompt"].strip()) > 0)
        except Exception as e:
            print(f"从ModelScope加载失败，尝试从HuggingFace加载: {e}")
            ds = load_dataset(dataset_name_or_path, split=split)

    # ensure prompt column
    if prompt_column != "prompt" and prompt_column in ds.column_names:
        ds = ds.rename_column(prompt_column, "prompt")

    return ds.select_columns(["prompt"]) if "prompt" in ds.column_names else get_default_dataset()


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training with combined rewards (TRL)")
    parser.add_argument("--model_name", type=str, default="/data/oss_bucket_0/Qwen3-32B")
    parser.add_argument("--reward_model_name", type=str, default="/data/oss_bucket_0/Qwen3-32B")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称或路径；为 None 时使用默认数据集 AI-ModelScope/COIG-CQIA",
    )
    parser.add_argument(
        "--subset_name",
        default=None,
        nargs="+",
        help="数据集子集名称（用于ModelScope），可以指定多个，用空格分隔。如果不指定，则使用所有默认子集",
    )
    parser.add_argument("--prompt_column", type=str, default="prompt", help="prompt列名（用于非ModelScope数据集）")
    parser.add_argument("--output_dir", type=str, default="./outputs-grpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=2, help="Samples per prompt for GRPO")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--rm_batch_size", type=int, default=32, help="Batch size for reward model inference")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Enable pin memory for dataloader")
    parser.add_argument("--use_liger_loss", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Override total training steps when dataset has no length",
    )
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

    # Determine torch dtype for model
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #   torch_dtype=torch_dtype,
        # device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    # Build reward function (only reward model)
    rm_fn = build_reward_model_fn(args.reward_model_name, torch_dtype=torch_dtype)

    # Store batch size for reward model in closure
    reward_batch_size = args.rm_batch_size

    # Adapter for TRL: GRPO expects either `reward_func` or `reward_funcs` depending on version; we provide one callable
    def reward_adapter(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        kwargs["batch_size"] = reward_batch_size
        return rm_fn(completions, prompts, **kwargs)

    # Data
    train_ds = load_prompts_dataset(
        dataset_name_or_path=args.dataset,
        subset_name=args.subset_name,
        split="train",
        prompt_column=args.prompt_column,
        tokenizer=tokenizer,  # 传递tokenizer以使用chat template
    )

    if len(train_ds) == 0:
        raise ValueError("训练数据集为空，请检查数据加载配置（dataset / subset_name 等）。")

    # 构建 GRPOConfig 参数字典，方便根据开关动态调整（与 PPO 版本风格保持一致）
    grpo_cfg_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_on_each_node": False,  # 只在主节点保存，避免多进程同时保存导致NCCL超时
        "eval_strategy": "no",
        "remove_unused_columns": False,
        # 为避免某些环境下 FP16 的 GradScaler 报错，这里禁用 fp16，推荐在支持的 GPU 上使用 bfloat16
        "fp16": False,
        "bf16": args.bf16,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "report_to": ["none"],
        "use_liger_loss": args.use_liger_loss,
        "push_to_hub": args.push_to_hub,
        "max_steps": args.max_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
    }

    # 如果使用8-bit优化器，则通过 optim 字段交给 TRL/transformers 选择 8bit 优化器
    if args.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb  # noqa: F401

            print("bitsandbytes已安装，可以在GRPO中使用8-bit优化器")
            # 默认使用 paged_adamw_8bit，与 SFT/PPO 部分保持一致
            grpo_cfg_kwargs["optim"] = "paged_adamw_8bit"
            print("GRPO 将使用 paged_adamw_8bit 优化器（可节省约50-75%优化器状态显存）")
        except ImportError:
            print("警告: 未安装bitsandbytes，8-bit优化器将无法使用")
            print("请运行: pip install bitsandbytes")
            print("将回退到标准优化器（adamw_torch）")

    grpo_cfg = GRPOConfig(**grpo_cfg_kwargs)

    # 创建自定义回调以处理分布式训练中的保存同步
    save_callback = DistributedSaveCallback()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_cfg,
        train_dataset=train_ds,
        reward_funcs=reward_adapter,
        callbacks=[save_callback],  # 添加自定义回调
    )

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
