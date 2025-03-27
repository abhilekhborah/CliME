import os
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/Climmeme/src")

from caq import ClimateAlignmentQuotient
from datasets import load_dataset
from dataclasses import dataclass
from groq import Groq
import json
import os
import time
from tqdm import tqdm

client = Groq(api_key="API_KEY")


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




if __name__ == "__main__":
    dataset = ClimmemeDataset("climedataset/CliME")
    caq = ClimateAlignmentQuotient()
    results = {}
    # Check if the result file exists
    output_file = "llama70_eval.json"
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
                response = client.chat.completions.create(
                    model="llama3-70b-8192",  # Or "llama-3.3-70b-Specdec" for speculative decoding
                    messages=[{"role": "user", "content": f"{value}"}],
                    temperature=0,
                    max_tokens=1024
                )
                response = response.choices[0].message.content
                caq_results = caq.calculate_caq(response)
                results[idx][key] = {
                    "llama_response": response,
                    "resonance": caq_results['component_scores']['detector'],
                    "articulation": caq_results['component_scores']['as'],
                    "transition": caq_results['component_scores']['transition_physical'],
                    "sentiment": caq_results['component_scores']['sentiment'],
                    "specificity": caq_results['component_scores']['specificity'],
                    "caq": caq_results['component_scores']['caq'],
                }
            output_file = "llama70_eval.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {output_file}")
            print(f"1 second sleep before next iteration")
            time.sleep(1)
        except Exception as e:
            print(f"An error occurred at index {idx}: {e}")
            sys.exit(1)


        

    






