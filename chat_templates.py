#!/usr/bin/env python3
"""Chat template system for preventing model collapse in benchmarks."""

import re
from pathlib import Path


def detect_model_family(model_path: str) -> str:
    """Detect model family from model path or name.
    
    Returns lowercase family name like 'gemma', 'qwen', 'llama', 'mistral', etc.
    """
    if not model_path:
        return "unknown"
    
    path_lower = model_path.lower()
    
    # Extract just the filename
    filename = Path(model_path).name.lower()
    
    # Check for model family indicators (order matters: check more specific first)
    if "gemma" in filename or "gemma" in path_lower:
        return "gemma"
    if "qwen" in filename or "qwen" in path_lower:
        return "qwen"
    if "nemotron" in filename or "nemotron" in path_lower:
        return "nemotron"
    if "llama" in filename or "llama" in path_lower:
        return "llama"
    if "mistral" in filename or "mistral" in path_lower:
        return "mistral"
    if "mixtral" in filename or "mixtral" in path_lower:
        return "mixtral"
    if "hermes" in filename or "hermes" in path_lower:
        return "hermes"
    if "phi" in filename or "phi" in path_lower:
        return "phi"
    
    return "unknown"


def apply_chat_template(prompt: str, model_family: str, system_prompt: str = None) -> str:
    """Apply model-specific chat template to prevent repetition loops.
    
    Chat templates format prompts in a way that models expect during generation,
    which prevents them from falling into repetitive loops when generating
    completions without proper instruction formatting.
    
    Args:
        prompt: The raw benchmark prompt
        model_family: Detected model family (gemma, qwen, llama, etc.)
        system_prompt: Optional system prompt to prepend
    
    Returns:
        Template-formatted prompt ready for generation
    """
    
    if model_family == "gemma":
        return _apply_gemma_template(prompt, system_prompt)
    elif model_family == "qwen":
        return _apply_qwen_template(prompt, system_prompt)
    elif model_family == "nemotron":
        return _apply_nemotron_template(prompt, system_prompt)
    elif model_family == "llama":
        return _apply_llama_template(prompt, system_prompt)
    elif model_family == "mistral":
        return _apply_mistral_template(prompt, system_prompt)
    elif model_family == "hermes":
        return _apply_hermes_template(prompt, system_prompt)
    else:
        # Unknown model - return prompt as-is
        return prompt


def _apply_gemma_template(prompt: str, system_prompt: str = None) -> str:
    """Gemma template format:
    
    <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    """
    result = "<start_of_turn>user\n"
    
    if system_prompt:
        result += f"{system_prompt}\n<end_of_turn>\n<start_of_turn>user\n{prompt}"
    else:
        result += prompt
    
    result += "\n<end_of_turn>\n<start_of_turn>model\n"
    return result


def _apply_qwen_template(prompt: str, system_prompt: str = None) -> str:
    """Qwen template format (varies by version, using common pattern):
    
    </s><|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    """
    result = "<|im_start|>user\n"
    
    if system_prompt:
        result += f"{system_prompt}\n<|im_end|>\n<|im_start|>user\n{prompt}"
    else:
        result += prompt
    
    result += "\n<|im_end|>\n<|im_start|>assistant\n"
    return result


def _apply_llama_template(prompt: str, system_prompt: str = None) -> str:
    """Llama 2/3 template format:
    
    [INST]{system_prompt if present else ""}{prompt}[/INST]
    """
    result = "[INST] "
    
    if system_prompt:
        result += f"{system_prompt} "
    
    result += f"{prompt} [/INST]"
    return result


def _apply_mistral_template(prompt: str, system_prompt: str = None) -> str:
    """Mistral template format (simple version):
    
    [INST]{prompt}[/INST]
    """
    result = "[INST] "
    
    if system_prompt:
        result += f"{system_prompt} "
    
    result += f"{prompt} [/INST]"
    return result


def _apply_nemotron_template(prompt: str, system_prompt: str = None) -> str:
    """Nemotron template format (NVIDIA conversational instruct model):
    
    <|user|>\n{prompt}\n</s>\n<|assistant|>
    
    Nemotron uses NVIDIA's proprietary <|user|>/<|assistant|> token format
    with </s> as the turn separator. This is required for proper generation
    and prevents the repetition loops seen without a template.
    """
    result = "<|user|>\n"
    
    if system_prompt:
        result += f"{system_prompt}\n</s>\n<|user|>\n{prompt}"
    else:
        result += prompt
    
    result += "\n</s>\n<|assistant|>"
    return result


def _apply_hermes_template(prompt: str, system_prompt: str = None) -> str:
    """Hermes template format (similar to Phi/Instruct models):
    
    <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    """
    result = "<|im_start|>user\n"
    result += prompt
    result += "\n<|im_end|>\n<|im_start|>assistant\n"
    return result
