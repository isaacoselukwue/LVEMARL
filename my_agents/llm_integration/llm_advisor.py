import json
import os
import torch
from transformers import pipeline, AutoTokenizer

class LLMAdvisor:
    """
    A wrapper for a local Hugging Face LLM to provide reward shaping scores.
    Uses the standard `transformers` library pipeline for inference.
    """
    def __init__(self, model_name, finetune_log_path='logs/finetune_data.jsonl'):
        """
        Initializes the LLM advisor using the transformers pipeline.

        Args:
            model_name (str): The Hugging Face model ID.
            finetune_log_path (str): Path to save data for future fine-tuning.
        """
        print(f"Initializing LLM Advisor with Hugging Face model: {model_name}...")
        try:
            if not torch.cuda.is_available():
                print("Warning: CUDA not available. LLM will run on CPU, which may be very slow.")
            
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True 
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            self.finetune_log_path = finetune_log_path
            os.makedirs(os.path.dirname(finetune_log_path), exist_ok=True)
            print("LLM Advisor initialized successfully.")
        except (ImportError, ValueError) as e:
            print(f"FATAL: Failed to initialize LLM Advisor. Error: {e}")
            print("\n--- TROUBLESHOOTING STEPS ---")
            print("1. Missing Library: Make sure you have installed all dependencies with 'pip install transformers torch accelerate bitsandbytes'.")
            print("2. Gated Model Access: For models like Llama, you must log in.")
            print("   - Step A: Go to the model's Hugging Face page and accept the terms.")
            print("   - Step B: Run 'huggingface-cli login' in your terminal and paste your access token.")
            print("---------------------------\n")
            self.pipe = None
        except Exception as e:
            print(f"FATAL: An unexpected error occurred during LLM initialization: {e}")
            self.pipe = None

    def get_reward_score(self, prompt):
        """
        Generates a response from the LLM and parses the score.
        """
        if not self.pipe:
            return 0.0, "{}"

        messages = [
            {"role": "system", "content": "You are a helpful and concise strategic advisor for a video game. Respond ONLY with the requested JSON object."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            # The tokenizer handles the specific chat template for the loaded model (Llama, Qwen, etc.)
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.pipe(
                formatted_prompt,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                eos_token_id=self.pipe.tokenizer.eos_token_id
            )
            assistant_response = outputs[0]["generated_text"][len(formatted_prompt):]

            json_start = assistant_response.find('{')
            json_end = assistant_response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                print(f"Warning: LLM response contained no JSON. Response: {assistant_response}")
                return 0.0, assistant_response

            json_str = assistant_response[json_start:json_end]
            parsed_json = json.loads(json_str)
            score = float(parsed_json.get("score", 0.0))
            score = max(-1.0, min(1.0, score))
            
            return score, assistant_response

        except Exception as e:
            print(f"Warning: An unexpected error occurred during LLM inference: {e}")
            return 0.0, "{}"

    def log_for_finetuning(self, trajectory_data):
        """
        Logs a successful trajectory for future fine-tuning.
        """
        try:
            with open(self.finetune_log_path, 'a') as f:
                for item in trajectory_data:
                    f.write(json.dumps(item) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to fine-tuning log file. Error: {e}")
