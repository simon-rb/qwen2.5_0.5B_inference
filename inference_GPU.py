import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define model name and prompt
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
prompt = "What is quantum mechanics in one paragraph?"

# Set the device to GPU (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **Optional: Enable 8-bit Quantization** (requires `bitsandbytes`)
# Uncomment this section to use quantization
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,  # Use FP16 for GPU
    device_map="auto" if device.type == "cuda" else None  # Efficient memory handling on GPU
    # quantization_config=quantization_config  # Uncomment for 8-bit quantization
)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize the input and move it to the device
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_length = inputs["input_ids"].shape[1]  # **Metric 1: Input Length**
print(f"Input length: {input_length} tokens")

# Measure inference time
print("Running inference...")
start_time = time.time()

# Synchronize to ensure accurate timing for GPU inference
if torch.cuda.is_available():
    torch.cuda.synchronize()

# Generate the output
outputs = model.generate(**inputs, max_new_tokens=512)

# Synchronize again to ensure accurate timing for GPU
if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.time()

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nResponse:\n", response)

# **Metric 2: Inference Time**
inference_time = end_time - start_time
print(f"\nInference time: {inference_time:.2f} seconds")

# **Metric 3: Speed (Tokens per Second)**
tokens_generated = outputs.shape[-1]  # Number of tokens generated
speed = tokens_generated / inference_time
print(f"Speed: {speed:.2f} tokens/sec")

# **Optional: Report GPU Memory Usage**
if device.type == "cuda":
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"GPU Memory Used: {gpu_memory:.2f} GB")
