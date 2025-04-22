# LaMoE
Scratch implementation of Sparse Mixture of Experts based Decoder model for text generation.

## Description:
***Mixture of Experts*** (MoE) is an ensemble learning paradigm introduced to scale the language model for handling large-scale data and to achieve higher accuracy.
The main idea of this technique is to introduce multiple specialized models to learn different subsets of the data using gating mechanism. This approach improves overall accuracy and efficiency by leveraging the strengths of specialized models.

The key difference between Vanilla Transformer and MoE based transformer is that,
In Vanilla transformer, there will be a dense feedforward NN that utilizes every parameters for learning all the data. 
In MoE based transformer, there will be mutiple feedforward NN that learns different parts of the data.

***Sparse Mixture of Experts*** (Sparse MoE) is a type of MoE, where only a subset of the model's experts, or specialized sub-models, are active for each input. This contrasts with dense MoEs, where all experts are used for all inputs. Sparse MoE reduces computational cost and memory usage by selectively activating a smaller number of experts based on the input, making it more efficient for scaling up large models. 

Key Components of a MoE Model:
* *Experts*:
These are feedforward neural networks trained on different parts of the data.
* *Gating Network*:
This network learns to predict which experts are most appropriate for a given input. It essentially selects a subset of experts to contribute to the final prediction. 
* *Output Aggregation*:
The outputs of the selected experts are combined to produce the final prediction. 

> Intution behind the name of the model -> Llama + MoE = LaMoE (Integrating MoE module into Llama 2 Architecture)

## Instructions:
All of these implementations are done using Conda. (Note: It requires python version = 3.10. Make sure of it.)
1. Clone this repository.

   ```
   git clone https://github.com/harishhirthi/NpyLinear.git
   cd NpyLinear
   ```
2. Create conda environment using environment.yml and activate this environment.
   ```
   conda env create -f environment.yml
   ``` 
3. To train lamoe using script. 
   ```
   cd scripts
   python train_eval.py
   ```
4. For inferencing using script.
   ```
   cd scripts
   python inference.py
   ```

## Datasets:
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

**Note:** *mlflow* is used for logging and tracking the training updates.

#### Screenshots:
These are the sample screenshots of training and inference using scripts.

Training:
![Screenshot (131)](https://github.com/user-attachments/assets/da6fef2a-0d3d-4e35-a240-7e5ffccdeb35)
![Screenshot (132)](https://github.com/user-attachments/assets/96fa6d49-f254-420e-a8a0-55b6a01a02be)

Inference:
![Screenshot (134)](https://github.com/user-attachments/assets/2e69a2b8-e603-4602-90d2-3ae56808c556)

Loss:
![Screenshot (133)](https://github.com/user-attachments/assets/dca724c5-f43c-4b91-b9bb-00a37b4e106a)

##### Sample output:
```
User: Science and Technology are
Generating Text , ...
Model:
Science and Technology are also used in the United States customary landowners and the Netherlands as well as the number of new classes dropped markedly with only a year with just two screens including Germany and Ireland in their first half of the year for the first time in the UK for the first time since the summer of 2003 the World Cup is held in
User: Once, upon a time
Generating Text , ...
Model:
Once upon a time scale it is a great number of times it is a positive correlation between the two points at which the observer is the time derivative of the electric field E is related to the field strength at the point where the permittivity is measured in radians per meter.
User: Physics
Generating Text , ....
Model:
Physics for example the second is the 367th greatest single quarter of the same period of 629 seconds long in 1572 seconds.
User: exit
```
*Note:* The above sample output is a result of initial training of the model for 1000 iterations, without any additional strategies like early stopping, learning rate scheduler etc. Also, this training setup begins to overfit from 300 iterations which has to be taken care. The important motive of this implementation is to get the glimpse of processing large corpus of text and using them to train the large language model that are scaled using MoE architecture.

## References:
I would like to credit these references for providing the valuable resources to make this implementation.
1. [MoE Visual Guide](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
2. [Hugging Face Blog](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch)
3. [Paper](https://arxiv.org/pdf/2401.04088.pdf), [Paper](https://arxiv.org/pdf/1701.06538.pdf), [Paper](https://arxiv.org/pdf/2101.03961)
4. [Mistral](https://github.com/mistralai/mistral-inference)
5. [DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3)
6. [GitHub](https://github.com/AviSoori1x/makeMoE), [GitHub](https://github.com/harishhirthi/Torch-LLaMA-Inference), [GitHub](https://github.com/davidmrau/mixture-of-experts).


## Github structure:
```
LaMoE
│   README.md
│   environment.yml    
└───lamoe/
│   │   __init__.py
│   │   config.py
│   │   moe.py
│   │   tokenizer.py
│   │   transformer.py
│   │   utils.py   
└───notebooks/
│   │   Dataset.ipynb
│   │   Inference.ipynb
│   │   Training.ipynb
└───scritps/
│   │   inference.py
│   │   train_eval.py
└───Data/
    |___Book.txt
```

<div align="center">
  <strong> !!!! Happy Learning !!!! </strong>
</div>
