import torch
import sys
import os
sys.path.append("Climmeme/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_llama_70b import ClimmemeDataset, Prompts
from caq import ClimateAlignmentQuotient
import json
from tqdm import tqdm
import argparse
class QwenModel:
    def __init__(self, model_name="Qwen/QwQ-32B-AWQ"):
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",  # Automatically distribute across available GPUs
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use half precision to reduce memory usage
        )

    def generate_response(self, prompt, max_length=256):
        # Tokenize the input
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate the response
        output = self.model.generate(
            **input_ids,
            max_new_tokens=max_length,
            temperature=0,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Decode and return the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt):]  # Return only the generated text, not the prompt

# Example usage
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Qwen model on climate dataset.")
    parser.add_argument("--file", type=str, required=True, help="Output file name.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing.")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing.")
    args = parser.parse_args()

    dataset = ClimmemeDataset("climedataset/CliME")
    caq = ClimateAlignmentQuotient()
    qwen_model = QwenModel()

    output_file = args.file
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)

    # Check if the result file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_results = json.load(f)
        # Start from the next index after the last available key if start_idx is not explicitly provided
        start_idx = max(start_idx, max(map(int, existing_results.keys())) + 1)
        results = existing_results
    else:
        results = {}

    print(f"start_idx = {start_idx}")
    print(f"end_idx = {end_idx}")
    # Data traversal
    for idx in tqdm(range(start_idx, end_idx), desc="Processing dataset"):
        text, description = dataset[idx]
        try:
            prompts = Prompts().with_description(description)

            results[idx] = {
                "text": text,
                "description": description,
            }
            for key, value in prompts.items():
                response = qwen_model.generate_response(value)
                caq_results = caq.calculate_caq(response)
                results[idx][key] = {
                    "qwen_response": response,
                    "resonance": caq_results['component_scores']['detector'],
                    "articulation": caq_results['component_scores']['as'],
                    "transition": caq_results['component_scores']['transition_physical'],
                    "sentiment": caq_results['component_scores']['sentiment'],
                    "specificity": caq_results['component_scores']['specificity'],
                    "caq": caq_results['component_scores']['caq'],
                }

            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error processing post {idx}: {e}")
            exit()

    print("Done")
