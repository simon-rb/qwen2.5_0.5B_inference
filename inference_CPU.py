import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define model name and prompt
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
prompt = "What is quantum mechanics in one paragraph?"

# Set the device to CPU
device = torch.device("cpu")
print("Using device: CPU")

# **Optional: Enable 8-bit Quantization**
# Uncomment to use quantization (requires `bitsandbytes` installed)
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model and tokenizer with FP32 precision or 8-bit quantization
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Change to FP32 or enable 8-bit quantization
    # quantization_config=quantization_config  # Uncomment for 8-bit quantization
)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize the input and measure input length
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_length = inputs["input_ids"].shape[1]  # **Metric 1: Input Length**
print(f"Input length: {input_length} tokens")

# Measure inference time
print("Running inference...")
start_time = time.time()

# Generate the output
outputs = model.generate(**inputs, max_new_tokens=150)

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