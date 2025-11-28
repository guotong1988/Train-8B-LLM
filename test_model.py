import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


def load_model_and_tokenizer(model_path: str, device: str = None, max_memory_MB: int = None, max_seq_length: int = None):
    """加载模型和分词器
    
    Args:
        model_path: 模型路径
        device: 设备（cuda/cpu），如果为None则自动检测
        max_memory_MB: 最大显存限制（MB），用于模型加载时的内存管理
        max_seq_length: 最大序列长度（可选，用于设置tokenizer.model_max_length）
    """
    print(f"加载模型: {model_path}")
    
    # 加载分词器（与train_sft.py保持一致）
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 设置 tokenizer 的 model_max_length（与train_sft.py保持一致）
    if max_seq_length is not None:
        try:
            current_max_length = getattr(tokenizer, 'model_max_length', None)
            if current_max_length is None or current_max_length > 1e10 or current_max_length > max_seq_length:
                tokenizer.model_max_length = max_seq_length
                print(f"设置 tokenizer.model_max_length 为 {max_seq_length}")
            else:
                print(f"tokenizer.model_max_length 已设置为 {current_max_length}")
        except Exception as e:
            print(f"警告: 无法设置 tokenizer.model_max_length: {e}")
    
    # 自动检测设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 配置模型加载参数
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    # 如果指定了显存限制且使用CUDA，使用max_memory参数
    if max_memory_MB is not None and device.startswith("cuda"):
        max_memory = {0: f"{max_memory_MB}MB"}
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
        print(f"使用显存限制: {max_memory_MB}MB")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # 如果未使用device_map，手动移动模型到设备
    if "device_map" not in model_kwargs:
        model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    # 清理显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    print("模型加载完成！")
    return model, tokenizer, device


def format_prompt(prompt: str, use_chat_template: bool = False, tokenizer=None):
    """格式化提示词（与SFT训练时的文本格式保持一致）
    
    Args:
        prompt: 原始提示词
        use_chat_template: 是否使用聊天模板（默认True，与训练时一致）
        tokenizer: 分词器（当use_chat_template=True时需要）
    """
    # 可选：使用chat template（如果你的训练数据本身就是chat模板格式）
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            # 构建messages列表（与训练时format_conversation中的格式一致）
            messages = [{"role": "user", "content": str(prompt)}]
            # 使用chat template，添加generation prompt（用于推理）
            # 注意：训练时使用 add_generation_prompt=False，推理时使用 add_generation_prompt=True
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # 推理时需要添加generation prompt
            )
            return formatted_prompt
        except Exception as e:
            print(f"警告: 使用chat template失败: {e}，使用简单格式")
            # 回退到简单格式（与train_sft.py中的简单格式保持一致）
            return f"用户: {prompt}\n助手: "
    else:
        # 默认：使用与 train_sft.py 中 format_dataset_for_sft 一致的简单格式
        # 格式：用户: {prompt}\n助手:
        if not prompt.startswith("用户:"):
            return f"用户: {prompt}\n助手: "
        # 如果已经包含"助手:"，直接返回
        return prompt


def find_assistant_start_position(input_ids, tokenizer):
    """找到助手回复开始位置（参考train_sft.py中的逻辑）
    
    Args:
        input_ids: 输入序列的token IDs（可以是完整的生成输出）
        tokenizer: 分词器
    
    Returns:
        assistant_start_idx: 助手回复开始位置的索引，如果找不到则返回None
    """
    # 检测助手回复标记token
    assistant_markers = ["助手:", "assistant:", "Assistant:", "GPT:", "gpt:"]
    assistant_token_ids = []
    for marker in assistant_markers:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if ids:
            assistant_token_ids.extend(ids)
    assistant_token_ids = list(set(assistant_token_ids))
    
    # 检测chat template的特殊token
    assistant_special_token_ids = []
    special_tokens_to_check = [
        "<|im_start|>assistant",
        "<|assistant|>",
        "<|im_start|>assistant\n",
        "assistant\n",
    ]
    for special_token in special_tokens_to_check:
        try:
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                token_id = tokenizer.convert_tokens_to_ids(special_token)
                if token_id != tokenizer.unk_token_id:
                    assistant_special_token_ids.append(token_id)
            ids = tokenizer.encode(special_token, add_special_tokens=False)
            if ids:
                assistant_special_token_ids.extend(ids)
        except:
            pass
    assistant_special_token_ids = list(set(assistant_special_token_ids))
    
    assistant_start_idx = None
    
    # 方法1: 查找助手回复标记token（如"助手:"、"assistant:"等）
    for marker_id in assistant_token_ids:
        positions = (input_ids == marker_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            # 找到最后一个标记（因为可能有多个，我们想要最后一个，即真正的助手回复开始）
            assistant_start_idx = positions[-1].item()
            # 跳过标记本身，从标记后的内容开始
            assistant_start_idx += 1
            return assistant_start_idx
    
    # 方法2: 查找chat template的特殊token
    if assistant_start_idx is None:
        for special_token_id in assistant_special_token_ids:
            positions = (input_ids == special_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                assistant_start_idx = positions[-1].item()
                assistant_start_idx += 1
                return assistant_start_idx
    
    # 方法3: 通过文本解码查找助手回复标记
    if assistant_start_idx is None:
        seq_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        assistant_keywords = [
            "<|im_start|>assistant",
            "<|assistant|>",
            "assistant\n",
            "Assistant\n",
            "助手:",
            "assistant:",
        ]
        for keyword in assistant_keywords:
            if keyword in seq_text:
                keyword_pos = seq_text.find(keyword)
                if keyword_pos >= 0:
                    prefix = seq_text[:keyword_pos + len(keyword)]
                    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                    if len(prefix_ids) < len(input_ids):
                        assistant_start_idx = len(prefix_ids)
                        return assistant_start_idx
    
    return assistant_start_idx


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    use_chat_template: bool = False,  # 默认False，与当前SFT脚本的简单“用户/助手”格式一致
    seed: int = None,
):
    """生成回复
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示词
        device: 设备
        max_new_tokens: 最大生成token数
        temperature: 温度参数（控制随机性）
        top_p: nucleus采样参数
        top_k: top-k采样参数
        do_sample: 是否使用采样
        use_chat_template: 是否使用聊天模板
        seed: 随机种子
    """
    if seed is not None:
        set_seed(seed)
    
    # 格式化提示词（与训练时格式保持一致）
    formatted_prompt = format_prompt(prompt, use_chat_template=use_chat_template, tokenizer=tokenizer)
    
    # 调试：打印格式化后的提示词（前200字符）
    if len(formatted_prompt) > 200:
        print(f"格式化后的提示词（前200字符）: {formatted_prompt[:200]}...")
    else:
        print(f"格式化后的提示词: {formatted_prompt}")
    
    # 编码输入
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    
    # 生成参数
    # 添加防止复读的参数
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,  # 添加重复惩罚，减少复读
    }
    
    # 生成回复
    print("正在生成回复...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )
    
    # 解码输出（参考train_sft.py中的-100逻辑，只提取助手回复部分）
    full_output_ids = outputs[0]
    
    # 先获取新生成的部分
    generated_ids = full_output_ids[input_length:]
    
    if len(generated_ids) == 0:
        # 如果没有生成任何内容，返回空字符串
        print("警告: 模型没有生成任何内容")
        return ""
    
    # 调试信息：打印新生成部分的原始解码（用于诊断）
    generated_text_raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"调试: 新生成部分（原始，前200字符）: {generated_text_raw[:200]}")
    print(f"调试: 新生成部分token数量: {len(generated_ids)}")
    
    # 检查是否复读了输入（复读检测）
    # 如果新生成的部分包含了输入提示词的关键部分，可能是复读
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    input_keywords = ["用户:", "user:", "User:", "<|im_start|>user", "<|user|>"]
    is_repeating = False
    for keyword in input_keywords:
        if keyword in input_text and keyword in generated_text_raw:
            # 检查是否在生成的部分中重复了输入的关键部分
            input_after_keyword = input_text.split(keyword)[-1][:50] if keyword in input_text else ""
            if input_after_keyword and input_after_keyword.strip() in generated_text_raw:
                is_repeating = True
                print(f"警告: 检测到可能的复读行为（包含输入关键词: {keyword}）")
                break
    
    # 在新生成的部分中查找助手标记（处理复读的情况）
    # 如果模型复读了输入，新生成的部分可能包含"用户: xxx\n助手: yyy"这样的格式
    # 如果模型正常生成，新生成的部分应该直接是助手回复内容（不包含助手标记）
    generated_assistant_idx = find_assistant_start_position(generated_ids, tokenizer)
    print(f"调试: find_assistant_start_position返回: {generated_assistant_idx}")
    
    # 如果检测到复读，强制查找助手标记
    if is_repeating and generated_assistant_idx is None:
        # 尝试更宽松的查找方法
        generated_text_lower = generated_text_raw.lower()
        assistant_keywords = ["助手:", "assistant:", "assistant\n", "<|im_start|>assistant", "<|assistant|>"]
        for keyword in assistant_keywords:
            if keyword.lower() in generated_text_lower:
                keyword_pos = generated_text_raw.lower().find(keyword.lower())
                if keyword_pos >= 0:
                    # 找到关键词位置，尝试定位token位置
                    prefix = generated_text_raw[:keyword_pos + len(keyword)]
                    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                    if len(prefix_ids) < len(generated_ids):
                        generated_assistant_idx = len(prefix_ids)
                        print(f"调试: 通过文本匹配找到助手标记位置: {generated_assistant_idx}")
                        break
    
    # 检查是否找到了助手标记，并且位置有效
    if generated_assistant_idx is not None and generated_assistant_idx < len(generated_ids):
        # 检查助手标记之后是否还有足够的内容（至少要有几个token，否则可能是误判）
        remaining_tokens = len(generated_ids) - generated_assistant_idx
        if remaining_tokens > 3:  # 如果助手标记之后还有至少3个token，才认为是有效的助手标记
            # 在新生成的部分中找到了助手标记，说明模型可能复读了输入
            # 只解码助手标记之后的内容（助手回复部分）
            print(f"调试: 在新生成部分中找到助手标记，位置: {generated_assistant_idx}, 剩余token数: {remaining_tokens}, 将解码位置 {generated_assistant_idx} 到 {len(generated_ids)}")
            response_ids = generated_ids[generated_assistant_idx:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            # 清理可能的残留标记
            response = response.strip()
            # 如果回复以助手标记开头，移除它
            for marker in ["助手:", "assistant:", "Assistant:"]:
                if response.startswith(marker):
                    response = response[len(marker):].strip()
            print(f"调试: 解码后的回复长度: {len(response)}, 内容（前200字符）: {response[:200] if len(response) > 200 else response}")
        else:
            # 助手标记之后内容太少，可能是误判，直接解码所有新生成的内容
            print(f"调试: 助手标记位置 {generated_assistant_idx} 之后内容太少（{remaining_tokens}个token），可能是误判，直接解码所有新生成内容")
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # 清理可能的残留标记
            response = response.strip()
            for marker in ["助手:", "assistant:", "Assistant:"]:
                if response.startswith(marker):
                    response = response[len(marker):].strip()
            print(f"调试: 解码后的回复长度: {len(response)}, 内容（前200字符）: {response[:200] if len(response) > 200 else response}")
    else:
        # 在新生成的部分中没找到助手标记，说明模型正常生成
        # 直接解码所有新生成的内容（这就是助手回复）
        print(f"调试: 未在新生成部分中找到助手标记，直接解码所有新生成内容")
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # 清理可能的残留标记和特殊token
        response = response.strip()
        # 移除可能的助手标记前缀
        for marker in ["助手:", "assistant:", "Assistant:"]:
            if response.startswith(marker):
                response = response[len(marker):].strip()
        # 移除可能的chat template标记
        for marker in ["<|im_start|>assistant", "<|assistant|>"]:
            if response.startswith(marker):
                response = response[len(marker):].strip()
        print(f"调试: 解码后的回复长度: {len(response)}, 内容（前200字符）: {response[:200] if len(response) > 200 else response}")
    
    return response.strip()


def interactive_mode(model, tokenizer, device, **generation_kwargs):
    """交互式模式"""
    print("\n进入交互模式（输入 'quit' 或 'exit' 退出）")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n用户: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["quit", "exit", "退出"]:
                print("再见！")
                break
            
            response = generate_response(
                model, tokenizer, prompt, device, **generation_kwargs
            )
            print(f"\n助手: {response}")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="测试模型 - 输入提示词，返回结果")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--prompt", type=str, default=None,
                        help="提示词（如果不提供，将进入交互模式）")
    parser.add_argument("--device", type=str, default=None,
                        help="设备（cuda/cpu），默认自动检测")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="温度参数（控制随机性，越高越随机）")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="nucleus采样参数")
    parser.add_argument("--top_k", type=int, default=50,
                        help="top-k采样参数")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="使用采样生成（默认启用）")
    parser.add_argument("--no_sample", action="store_true",
                        help="禁用采样（使用贪婪解码）")
    parser.add_argument("--use_chat_template", action="store_true", default=False,
                        help="使用模型的chat template格式化提示词（仅当你的训练数据本身也是chat模板时才建议开启）")
    parser.add_argument("--no_chat_template", action="store_true",
                        help="禁用chat template，使用简单格式")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="最大序列长度（可选，用于设置tokenizer.model_max_length，默认使用tokenizer的默认值）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--max_memory_MB", type=int, default=None,
                        help="每个GPU的最大显存限制（MB），用于模型加载时的内存管理")
    parser.add_argument("--output_file", type=str, default=None,
                        help="将结果保存到文件（仅单次推理模式）")
    
    args = parser.parse_args()
    
    # 处理采样参数
    do_sample = args.do_sample and not args.no_sample
    
    # 处理chat template参数（默认关闭；仅当训练数据本身使用chat模板时才建议开启）
    use_chat_template = args.use_chat_template and not args.no_chat_template
    if use_chat_template:
        print("将使用chat template格式化提示词（推荐，确保训练和推理格式一致）")
    else:
        print("警告: 未使用chat template，使用简单格式")
    
    # 加载模型和分词器（与train_sft.py保持一致）
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        max_memory_MB=args.max_memory_MB,
        max_seq_length=args.max_seq_length,
    )
    
    # 检查模型是否支持chat template
    if use_chat_template and not hasattr(tokenizer, "apply_chat_template"):
        print("警告: 分词器不支持chat template，将回退到简单格式")
        use_chat_template = False
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": do_sample,
        "use_chat_template": use_chat_template,
        "seed": args.seed,
    }
    
    # 如果提供了提示词，执行单次推理
    if args.prompt:
        print(f"\n提示词: {args.prompt}")
        response = generate_response(
            model, tokenizer, args.prompt, device, **generation_kwargs
        )
        print(f"\n回复: {response}")
        
        # 如果指定了输出文件，保存结果
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(f"提示词: {args.prompt}\n\n")
                f.write(f"回复: {response}\n")
            print(f"\n结果已保存到: {args.output_file}")
    else:
        # 进入交互模式
        interactive_mode(model, tokenizer, device, **generation_kwargs)


if __name__ == "__main__":
    main()

