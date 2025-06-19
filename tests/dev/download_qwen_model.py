#!/usr/bin/env python3
"""Download Qwen3-0.6B model from Hugging Face"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = "Qwen/Qwen3-0.6B"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print("Downloading {} model...".format(model_id))
print("Cache directory: {}".format(cache_dir))

# Download tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

# Download model
print("Downloading model weights and config...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    cache_dir=cache_dir,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("✅ Model downloaded successfully!")
print("Model location: {}".format(model.config._name_or_path))

# Get the actual path where the model is stored
from transformers.utils import cached_file
config_path = cached_file(model_id, "config.json", cache_dir=cache_dir)
print("Config file: {}".format(config_path))
model_dir = os.path.dirname(config_path)
print("Model directory: {}".format(model_dir))