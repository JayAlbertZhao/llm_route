import json
import os
import random
from typing import List, Dict, Generator
from datasets import load_dataset
from transformers import AutoTokenizer

class WorkloadProcessor:
    def __init__(self, dataset_name="allenai/WildChat-1M", output_path="data/processed_workload.jsonl", tokenizer_name="gpt2"):
        self.dataset_name = dataset_name
        self.output_path = output_path
        # Suppress parallelism warning if needed
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"Error loading tokenizer {tokenizer_name}: {e}")
            # Fallback or re-raise
            raise e
            
        # Buckets: (min, max) -> list of prompts
        self.buckets = {
            "short": (0, 128),
            "medium": (128, 512),
            "long": (512, 1024),
            "extra_long": (1024, 8192) 
        }
        self.data_buckets = {k: [] for k in self.buckets}

    def process(self, max_samples=50000):
        print(f"Loading dataset {self.dataset_name} (streaming)...")
        # Use streaming=True to avoid huge downloads
        try:
            dataset = load_dataset(self.dataset_name, split="train", streaming=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        count = 0
        processed_count = 0
        
        for sample in dataset:
            if processed_count >= max_samples:
                break
                
            # Extract conversation
            conversation = sample.get("conversation")
            if not conversation:
                continue
                
            # Find first user message
            user_msg = None
            for msg in conversation:
                if msg.get("role") == "user":
                    user_msg = msg.get("content")
                    break
            
            if user_msg and isinstance(user_msg, str) and len(user_msg.strip()) > 0:
                try:
                    tokens = self.tokenizer.encode(user_msg)
                    length = len(tokens)
                    
                    # Bucket
                    for bucket_name, (min_len, max_len) in self.buckets.items():
                        if min_len <= length < max_len:
                            self.data_buckets[bucket_name].append({
                                "prompt": user_msg,
                                "token_len": length,
                                "dataset_id": sample.get("conversation_hash", count) # Use hash or counter
                            })
                            processed_count += 1
                            if processed_count % 1000 == 0:
                                print(f"Processed {processed_count} samples...")
                            break
                except Exception as e:
                    pass # Skip problematic tokenization
            
            count += 1
        
        self.save()

    def save(self):
        print(f"Saving to {self.output_path}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        total_saved = 0
        with open(self.output_path, "w", encoding="utf-8") as f:
            for bucket, items in self.data_buckets.items():
                print(f"Bucket {bucket}: {len(items)} samples")
                for item in items:
                    item["bucket"] = bucket
                    f.write(json.dumps(item) + "\n")
                    total_saved += 1
        print(f"Total saved: {total_saved}")

class WorkloadLoader:
    def __init__(self, path="data/processed_workload.jsonl"):
        self.path = path
        self.data = []

    def load(self) -> List[Dict]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Workload file not found at {self.path}")
        
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        return self.data
    
    def get_iterator(self, loop=True) -> Generator[Dict, None, None]:
        if not self.data:
            self.load()
        
        while True:
            random.shuffle(self.data)
            for item in self.data:
                yield item
            if not loop:
                break

if __name__ == "__main__":
    # Example usage: python src/client/workload.py
    processor = WorkloadProcessor()
    processor.process(max_samples=10000)


