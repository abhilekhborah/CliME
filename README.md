# CliME (ACL '25 NLP4PI) - Outstanding Paper Award

This repository provides a complete framework for climate-related data creation, content description, evaluation of LLM responses, and the Climate Alignment Index (CAQ).

- **Dataset**: [CliME](https://huggingface.co/datasets/climedataset/CliME)

## Data Creation

Scripts for data creation are available in [`src/data_creation`](src/data_creation/). They include:
- `reddit_scrapping.py`
- `tweet_scraping.py`

These scripts are designed to scrape and process climate-related data from various sources.

## Content Description

### Janus Pro (VLM)

Located in [`VLMs/Janus`](VLMs/Janus), Janus Pro is used to generate visual-linguistic features for detailed descriptions of climate imagery.

**Installation:**
```bash
cd VLMs/Janus
pip install -e .
```

### Description Generation

Generate summaries by combining image and text data using the script:
`generate_summary_from_img_text.py`

- Loads the Clime Dataset from the Huggingface, processes and saves in the local disk. 

**Usage:**
```bash
python src/content_description/generate_summary_from_img_text.py
```

## Climate Alignment Index (CAQ)
CAQ routines and evaluations are integrated into the repository (see `caq.py` in the root directory for CAQ-related functionalities).

## Generating LLM Responses

Evaluate LLM responses using the generated summaries and the CAQ. Evaluation scripts are provided in `src/evaluate_llms_on_climate_desc`. For example, to run an evaluation using Gemini Flash2, execute:

```bash
python src/evaluate_llms_on_climate_desc/eval_gemini_flash2.py
```

Other available evaluation scripts include:
- `eval_llama_70b.py`
- `eval_qwen_qwq_32b.py`
- `eval_sonnet_37.py`
- `gpt_eval.py`

## Project Structure

```
├── src
│   ├── content_description
│   │   └── generate_summary_from_img_text.py
│   ├── data_creation
│   │   ├── reddit_scrapping.py
│   │   └── tweet_scraping.py
│   ├── evaluate_llms_on_climate_desc
│   │   ├── eval_gemini_flash2.py
│   │   ├── eval_llama_70b.py
│   │   ├── eval_qwen_qwq_32b.py
│   │   ├── eval_sonnet_37.py
│   │   └── gpt_eval.py
│   ├── cpaq.py            # CAQ related routines
│   ├── plot_3d.py
│   └── plot_gen.py
├── VLMs
│   └── Janus
│       ├── setup.py
│       └── ...
├── .gitignore
├── .gitmodules
├── LICENSE
└── README.md
```
