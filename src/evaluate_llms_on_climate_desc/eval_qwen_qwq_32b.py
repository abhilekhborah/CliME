import torch
import sys
import os
sys.path.append("/scratch/user/hasnat.md.abdullah/Climmeme/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_llama_70b import ClimmemeDataset, Prompts
from caq import ClimateAlignmentQuotient
import json
from tqdm import tqdm
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
    dataset = ClimmemeDataset("climedataset/CliME")
    caq = ClimateAlignmentQuotient()
    qwen_model = QwenModel()
     # Check if the result file exists
    output_file = "qwen_qwq_32B_eval.json"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_results = json.load(f)
        # Start from the next index after the last available key
        start_idx = max(map(int, existing_results.keys())) + 1
        results = existing_results
    else:
        # Start from the beginning of the dataset
        start_idx = 0
        results = {}
    #data traversal
    for idx in tqdm(range(start_idx, len(dataset)), desc="Processing dataset"):
        text, description = dataset[idx]
        try:
            prompts = Prompts().with_description(description)

            axioms = {}
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
            output_file = "qwen_qwq_32B_eval.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error processing post {idx}: {e}")
            exit()
    print("Done")
    









    # prompt = "Explain how neural networks work in simple terms:"
    # response = qwen_model.generate_response(prompt)
    # print(response)