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
import anthropic
import time



CLAUDE_API_KEY="API-KEY"
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
def get_claude_response(prompt_text, max_retries=3):
    """
    Get a response from GPT-4o with retry logic for rate limiting
    """
    retries = 0
    while retries < max_retries:
        try:
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=256,
                temperature=0.1,
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt_text}]
            )

 
            return response.content[0].text
        except Exception as e:
            retries += 1
            error_message = str(e)
            print(f"Error (attempt {retries}/{max_retries}): {error_message}")
            
            # Check if it's a rate limit error
            if "429" in error_message or "rate limit" in error_message.lower():
                wait_time = 60  # Wait 60 seconds before retrying
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                # For other errors, wait a shorter time
                print(f"Unexpected error. Waiting 10 seconds before retrying...")
                time.sleep(10)
                
            # If we've exhausted all retries, raise the exception
            if retries >= max_retries:
                print(f"Failed after {max_retries} attempts. Moving on.")
                return f"Error generating response: {error_message}"
def main():
    dataset = ClimmemeDataset("climedataset/CliME")
    caq =  ClimateAlignmentQuotient()

    results = {}
    output_file = "claude_sonnet_37_eval.json"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_results = json.load(f)
        # Start from the next index after the last available key if start_idx is not explicitly provided
        start_idx = max(map(int, existing_results.keys())) + 1
        results = existing_results
    else: 
        # Start from the beginning of the dataset or the provided start_idx
        start_idx = 0
        results = {}

    # Data traversal
    for idx in tqdm (range(start_idx, len(dataset)), desc="Processing dataset"):
        text, description = dataset[idx]
        try:
            prompts = Prompts().with_description(description)
            results[idx] = {
                "text": text,
                "description": description,
            }
            for key, value in prompts.items():
                response = get_claude_response(value)
                caq_results = caq.calculate_caq(response)
                results[idx][key] = {
                    "claude_response": response,
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
            # break
        except Exception as e:
            print(f"Error processing post {idx}: {e}")
            exit()

if __name__ == "__main__":
 
    main()
