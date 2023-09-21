# ADAPTING THE ADAPTERS FOR CODE-SWITCHING IN MULTILINGUAL ASR

## Improving performance of [Meta AI's MMS](https://arxiv.org/abs/2305.13516) in code-switching.
*Atharva Kulkarni, Ajinkya Kulkarni, Miguel Couceiro, Hanan Aldarmaki*

### **ABSTRACT**

Recently, large pre-trained multilingual speech models
have shown potential in scaling Automatic Speech Recogni-
tion (ASR) to many low-resource languages. Some of these
models employ language adapters in their formulation, which
helps to improve monolingual performance and avoids some
of the drawbacks of multi-lingual modeling on resource-rich
languages. However, this formulation restricts the usability
of these models on code-switched speech, where two lan-
guages are mixed together in the same utterance. In this
work, we propose ways to effectively fine-tune such mod-
els on code-switched speech, by assimilating information
from both language adapters at each language adaptation
point in the network. We also model code-switching as a
sequence of latent binary sequences that can be used to guide
the flow of information from each language adapter at the
frame level. The proposed approaches are evaluated on three
code-switched datasets encompassing Arabic, Mandarin, and
Hindi languages paired with English, showing consistent im-
provements in code-switching performance with at least 10%
absolute reduction in CER across all test sets.



### Brief description of our approaches
We modify the Wav2Vec2 transformer blocks used in MMS to use 2 pretrained adapter modules corresponding to the matrix and embedded languages to incorporate information from both. Based on this modification, we propose two code-switching approaches:

![image](https://github.com/Atharva7K/Multilingual-Chat-Room/assets/61614635/21e65a38-04b2-47e6-986b-1ee6487f8ab7)


#### 1) Post Adapter Switching
We add a Post-Adapyter-Code-Switcher network (PACS) inside every transformer block after the 2 adapter modules (see Figure 1a) . Output from the adapter modules is concatenated and fed to PACS which learns to assimilate information from both. The base model and the 2 pretrained adapter modules are kept frozen during the training hence only PACS and the output layer is trainable. PACS follows the same architectures as the adapter modules used in MMS: two feedforward layers with a LayerNorm layer and a linear
projection to 16 dimensions with ReLU activation

#### 2) Transformer Code Switching
We use a transformer network with sigmoid  as output activation as a Transformer Code Switcher (TCS). It learns to predict a code-switch-sequence O <sub>CS</sub> using output of the Wav2Vec2 Feature Projection block (Figure 1b). The code-switch-sequence is a latent binary sequence that helps to identify code-switching boundaries at frame level. It regulates the flow of information from two adapters to enable the network to handle code-switched speech by dynamically masking out one of the languages as per the switching equation :

![image](https://github.com/Atharva7K/Multilingual-Chat-Room/assets/61614635/39e58c62-e346-45b5-81fc-a53a236fd791)

We use  a threshold value of 0.5 to the output of the sigmoid activation to create binarized latent codes O <sub>CS</sub>. The base model and adapter are kept frozen, only TCS and the output layers are trained on code-switched data.

### Usage 

### Installation
Clone this repository 

```bash
git clone https://github.com/Atharva7K/MMS-Code-Switching
```
NOTE: This repo includes the entire codebase of [hugging face transformers](https://github.com/huggingface/transformers). We write our modifications on top of their codebase. Most of our modified code is in [this file](https://github.com/Atharva7K/MMS-Code-Switching/blob/main/transformers/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L926). 

#### Install dependancies

First we recommend creating a new conda environment especially if you have transformers already installed. We will be installing modified code for the transformers library from this repo which can cause conflicts with your existing installation. Hence create and activate new environment using 
```bash
conda create -n mms-code-switching python=3.10.2
conda activate mms-code-switching 
```

#### Install modified transformers code
```bash
cd transformers/
pip install -e .
```

#### Install other dependancies
```bash
pip install -r requirements.txt
```

#### Download model checkpoints:

| Model                | ASCEND (MER / CER) | ESCWA (WER / CER) | MUCS (WER / CER) | 
|----------------------|--------------------|--------------------|-------------------|
| **MMS with single language adapter:** |           |            |                  |               
| English              | 98.02 / 87.85   | 92.73 / 71.14    | 101.72 / 74.02 |  
| Matrix-language      | 71.98 / 66.76   | 75.98 / 46.38    | 58.05 / 49.20  |  
| **Proposed models for fine-tuning:** |           |            |                  |               
| Matrix-language-FT   | 45.97 / 44.13   [Download](https://zenodo.org/api/files/df69f0da-8c98-4f13-ac9b-b5469bee6928/ascend_finetuned_pytorch_model.bin)   | 77.47 / 37.69   [Download](https://zenodo.org/api/files/df69f0da-8c98-4f13-ac9b-b5469bee6928/qasr_finetuned_pytorch_model.bin)    | 66.19 / 41.10  [Download](https://zenodo.org/api/files/df69f0da-8c98-4f13-ac9b-b5469bee6928/mucs_finetuned_pytorch_model.bin)   | 
| Post Adapter Code Switching                 | 44.41 / 40.24   [Download](https://zenodo.org/api/files/df69f0da-8c98-4f13-ac9b-b5469bee6928/pacs_ascend_pytorch_model.bin)   | 75.50 / 46.69   [Download](#)    | 63.32 / 42.66   [Download](https://drive.google.com/file/d/1TjuIyugkKlW9_GiJU9vBV2SuLb-pRWfL/view?usp=drive_link)  | 
| Transformer Code Switching                  | 41.07 / 37.89   [Download](https://drive.google.com/file/d/1LzKnsYXvE1vImZj7TWkTGAxKJqBnMPN1/view?usp=drive_link)   | 74.42 / 35.54   [Download](https://drive.google.com/file/d/1hE9Cy3qo5XbEE3p1Lr1i3sTgfD6muGKp/view?usp=drive_link)    | 57.95 / 38.26  [Download](https://drive.google.com/file/d/1qs9cWSzNtFpA3Grqu_YoQl0c1uj1WvyI/view?usp=drive_link)   | 

We also provide MMS checkpoints after finetuning matrix-language adapters on the 3 datasets. NOTE: In order to do inference on these finetuned checkpoints, one should use standard implementation of [MMS from huggingface](https://huggingface.co/facebook/mms-1b-all) instead of our modified transformers code. 

#### Do inference

Use `main` branch for Transformer Code Switching (TCS) and `post-adapter-switching` branch for Post Adapter Code Swtiching (PACS).

Check `demo.ipynb` for inference demo.

#### Output transcripts

We also share transcripts generated by our proposed systems on the 3 datasets in `generated_transcripts/`. 
