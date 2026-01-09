import json
import os
import random

def create_dummy_data(output_path="data/processed_workload.jsonl", count=100):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a poem about a robot.",
        "What is the capital of France?",
        "Translate 'Hello world' to Python code.",
        "Summarize the history of the internet.",
        "How do I bake a cake?",
        "Define 'recursion'.",
        "Who won the 2022 World Cup?",
        "Write a short story about space travel.",
        "What is the meaning of life?"
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(count):
            prompt = random.choice(prompts)
            item = {
                "prompt": prompt,
                "token_len": len(prompt.split()) * 1.3, # rough estimate
                "dataset_id": f"dummy_{i}",
                "bucket": "short" if len(prompt) < 50 else "medium"
            }
            f.write(json.dumps(item) + "\n")
    
    print(f"Created {count} dummy samples in {output_path}")

if __name__ == "__main__":
    create_dummy_data()


