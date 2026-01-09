from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

MODEL_PATH = "./models/Qwen/Qwen3-8B"

def check_model():
    print(f"Checking model integrity at {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model directory does not exist.")
        return

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    try:
        print("Loading model weights (this may take a minute)...")
        # Load with low_cpu_mem_usage to avoid RAM spike, using torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            device_map="auto" # Will try to use GPU if available, or CPU
        )
        print("✅ Model weights loaded successfully.")
        
        # Simple generation test
        print("Running simple generation test...")
        inputs = tokenizer("Hello, are you working?", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation output: {response}")
        print("🎉 Model integrity check PASSED!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: If it's a safetensors error, try re-downloading.")

if __name__ == "__main__":
    check_model()

