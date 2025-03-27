# CliME

## 1.0 Data Creation
- Dataset link: [CliME](https://huggingface.co/datasets/climedataset/CliME)
- scripts can be found on [src/data_creation](src/data_creation/)

## 2.0 content description
### 2.1 Module: Janus Pro (VLM)
- path: [VLMs/Janus](VLMs/Janus)
- Install related packages 
    ```bash  
        cd VLMs/Janus
        pip install -e .
    ```
### 2.2 Description Generation
- run [generate_summary_from_img_text.py](src/content_description/generate_summary_from_img_text.py)

### 3.0 Generate LLM Responses on Descriptions
- 