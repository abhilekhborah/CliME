import os
import sys
import pickle
import time
sys.path.append("Climmeme/src")

from caq import ClimateAlignmentQuotient
from datasets import load_dataset
from dataclasses import dataclass
import json
from openai import OpenAI
from tqdm import tqdm

# Configure OpenAI with your API key
OPENAI_API_KEY = "API-KEY"  # Replace with your actual API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Checkpoint file path
CHECKPOINT_FILE = 'gpt4o_eval_checkpoint.pkl'
OUTPUT_FILE = 'gpt4o_eval.json'

class ClimmemeDataset:
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)
    
    def __getitem__(self, idx):
        row = self.dataset['train'][idx]
        text = row['text']
        description = row['description']
        return text, description

    def __len__(self):
        return len(self.dataset['train'])

@dataclass
class Prompts:
    actionability: str = "Instruction:\nAnalyze the climate-related message in the above description through an actionability lens. Respond in one unified paragraph that summarizes the key climate issues, identifies actionable solutions, evaluates their feasibility (high/medium/low), assesses explicit commitments (who, what, when, how), and highlights risks or unaddressed challenges. Do not output any extra information other than this analysis in your response."

    criticality: str = "Instruction:\nCritically evaluate the climate-related message in the above description. Respond in one unified paragraph that identifies core claims, assesses evidence quality, highlights unsubstantiated claims or oversimplified arguments, evaluates engagement with competing viewpoints, and analyzes its impact on climate discourse. Do not output any extra information other than this analysis in your response."

    justice: str = "Instruction:\nAnalyze the climate-related message in the above description through a justice lens. Respond in one unified paragraph that identifies centered/absent communities, assesses distribution of responsibility, evaluates acknowledgment of historical power imbalances, examines impacts on marginalized groups, and considers inclusion of cultural contexts. Do not output any extra information other than this analysis in your response."

    def with_description(self, description: str):
        return {
            "actionability": f"""Description: {description}\n{self.actionability}""",
            "criticality": f"""Description: {description}\n{self.criticality}""",
            "justice": f"""Description: {description}\n{self.justice}""",
        }

def get_gpt_response(prompt_text, max_retries=3):
    """
    Get a response from GPT-4o with retry logic for rate limiting
    """
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content
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

def load_checkpoint():
    """Load checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Resuming from checkpoint. Starting at index {checkpoint['current_index']}")
            return checkpoint['current_index'], checkpoint['results']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, {}
    else:
        print("No checkpoint found. Starting from the beginning.")
        return 0, {}

def save_checkpoint(current_index, results):
    """Save the current state to a checkpoint file"""
    checkpoint = {
        'current_index': current_index,
        'results': results
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at index {current_index}")

def save_results(results):
    """Save the results to a JSON file"""
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {OUTPUT_FILE}")

def main():
    # Initialize the dataset and CAQ calculator
    dataset = ClimmemeDataset("climedataset/CliME")
    caq = ClimateAlignmentQuotient()
    
    # Load checkpoint if it exists
    start_idx, results = load_checkpoint()
    
    # Process the dataset
    try:
        for idx in tqdm(range(start_idx, len(dataset)), desc="Processing dataset"):
            text, description = dataset[idx]
            
            try:
                prompts = Prompts().with_description(description)
                
                results[str(idx)] = {
                    "text": text,
                    "description": description,
                }
                
                # Process each prompt type
                for key, value in prompts.items():
                    # Get response from GPT-4o
                    response = get_gpt_response(value)
                    
                    # Calculate CAQ scores
                    caq_results = caq.calculate_caq(response)
                    
                    # Store results
                    results[str(idx)][key] = {
                        "gpt_response": response,
                        "resonance": caq_results['component_scores']['detector'],
                        "articulation": caq_results['component_scores']['as'],
                        "transition": caq_results['component_scores']['transition_physical'],
                        "sentiment": caq_results['component_scores']['sentiment'],
                        "specificity": caq_results['component_scores']['specificity'],
                        "caq": caq_results['component_scores']['caq'],
                    }
                
                # Save results after each item
                save_results(results)
                
                # Update checkpoint after each item
                save_checkpoint(idx + 1, results)
                
                # Short pause to avoid overwhelming the API
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing item at index {idx}: {e}")
                # Save checkpoint before exiting on error
                save_checkpoint(idx, results)
                # Continue with the next item instead of exiting
                continue
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        # Save checkpoint when interrupted
        save_checkpoint(idx, results)
        
    print("Evaluation complete!")

if __name__ == "__main__":
    main()