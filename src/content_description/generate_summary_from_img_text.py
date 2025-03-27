import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/Climmeme/VLMs/Janus")
from datasets import load_dataset, Dataset
from PIL import Image
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from tqdm import tqdm
import os

def print_gpu_memory(stage):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"{stage} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

class ClimmemeDataset:
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)
    
    def __getitem__(self, idx):
        row = self.dataset['train'][idx]
        text = row['text']
        likes = row['likes']
        image = row['image']
        return text, likes, image

    def __len__(self):
        return len(self.dataset['train'])
class ClimateMessageInferer:
    def __init__(self, model_path):
        self.config = AutoConfig.from_pretrained(model_path)
        self.language_config = self.config.language_config
        self.language_config._attn_implementation = 'eager'
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                                            language_config=self.language_config,
                                                            trust_remote_code=True)
        if torch.cuda.is_available():
            self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda()
        else:
            self.vl_gpt = self.vl_gpt.to(torch.float16)

        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def infer_climate_message(self, text, img):
        torch.cuda.empty_cache()

        question = f"""
        Caption: {text}

        Given the image and caption of the twitter/reddit post, please describe the climate-related message in this post, emphasizing actionable solutions. Include specific commitments or risks.
        """
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [img],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # print(f"Answer: {answer}")
        return answer

if __name__ == "__main__":
    # Create an instance of the dataset
    meme_dataset = ClimmemeDataset("abhilekhborah/climmeme")

    # specify the path to the model
    model_path = "deepseek-ai/Janus-Pro-7B"

    # List to store the updated data
    updated_data = []

    # Set to keep track of seen texts
    seen_texts = set()

    # Create an instance of the ClimateMessageInferer
    infer_climate_message = ClimateMessageInferer(model_path)
    # Iterate through the dataset and infer
    for idx in tqdm(range(len(meme_dataset)), desc="Processing posts"):
        try: 
            text, likes, og_img = meme_dataset[idx]
            # Check for duplicate text
            if text in seen_texts:
                continue
            
            seen_texts.add(text)
            og_img.convert("RGB")
            og_img.save("temp.jpg")
            img = "temp.jpg"
            answer =  infer_climate_message.infer_climate_message(text, img)
            # print(f"Post {idx} - Answer: {answer}")
            updated_data.append({"text": text, "likes": likes, "image": og_img, "description": answer})

        except Exception as e:
            print(f"Error processing post {idx}: {e}")
            continue
    # Create a new dataset with the updated data
    updated_dataset = Dataset.from_list(updated_data)
    # Save the updated dataset to a local file
    updated_dataset.save_to_disk("/scratch/user/hasnat.md.abdullah/Climmeme/updated_dataset")
    # load it and print it
    updated_dataset = Dataset.load_from_disk("/scratch/user/hasnat.md.abdullah/Climmeme/updated_dataset")

    print(updated_dataset)
    print(f"first row: {updated_dataset[0]}")

    # # Push the updated dataset to the hub
    # updated_dataset.push_to_hub("abhilekhborah/climmeme", token="YOUR_HUB_TOKEN")
