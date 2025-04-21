# LaMoE
Scratch implementation of Sparse MoE based Transformer model for text generation.

## Description:


## Instructions:
All of these implementations are done using Conda. (Note: It requires python version = 3.10. Make sure of it.)
1. Clone this repository

   ```
   git clone https://github.com/harishhirthi/NpyLinear.git
   cd NpyLinear
   ```
2. Create conda environment using environment.yml and activate this environment.
   ```
   conda env create -f environment.yml
   ``` 
3. To train lamoe, 
   ```
   cd scripts
   python train_eval.py
   ```
4. For inferencing 
   ```
   python inference.py
   ```

## Dataset:
1. [BBC News Summary](https://www.kaggle.com/datasets/pariza/bbc-news-summary)
2. [Avengers Dialogue](https://www.kaggle.com/datasets/divaxshah/avengers-and-iron-man-movies-dataset)
3. [Shakespeare Plays](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays)
4. [Physics Pages](https://www.kaggle.com/datasets/judehunt23/llm-science-exam-training-data-wiki-pages/data)
5. [Wiki Pages](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-english/data)

#### Information about repo:
This repo contains both notebook and script versions for training and inference.

1. lamoe - This folder contains source code for tokenizer and entirety of transformer model.
2. Dataset.ipynb - This notebook contains dataset processing, creating tokenizer and creation of tokens.
3. Training.ipynb - This notebook contains training of lamoe.
4. Inference.ipynb - This notebook contains the inference using trained lamoe.
5. train_eval.py - The native python script for training lamoe using command line.
6. inference.py - The native python script for inferencing.

Note: mlflow is used for logging and tracking the training updates.


## References:
I would like to credit these references for providing the valuable resources to make this implementation.
1. [MoE Visual Guide](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
2. [Hugging Face Blog](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch)
3. [Paper](https://arxiv.org/pdf/2401.04088.pdf), [Paper](https://arxiv.org/pdf/1701.06538.pdf), [Paper](https://arxiv.org/pdf/2101.03961)
4. [Mistral](https://github.com/mistralai/mistral-inference)
5. [DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3)
6. [GitHub](https://github.com/AviSoori1x/makeMoE), [GitHub](https://github.com/harishhirthi/Torch-LLaMA-Inference), [GitHub](https://github.com/davidmrau/mixture-of-experts).
