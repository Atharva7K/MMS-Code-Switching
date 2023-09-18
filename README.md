# ADAPTING THE ADAPTERS FOR CODE-SWITCHING IN MULTILINGUAL ASR

## Improving performance of Meta AI's MMS in code-switched setting. For more details, check our paper [here]
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

### Installation
Clone this repository 

```bash
git clone https://github.com/Atharva7K/MMS-Code-Switching
```
NOTE: This repo includes the entire codebase of [hugging face transformers](https://github.com/huggingface/transformers). We write our modifications on top of their codebase. Most of our modified code is in [this file](https://github.com/Atharva7K/MMS-Code-Switching/transformers/models/wav2vec2/modeling_wav2vec2.py). 

#### Install dependancies

First we recommend creating a new conda environment especially if you have transformers already installed. We will be installing modified code for the transformers library from this repo which can cause discrepenceis with your existing installation. Hence create new environment using 
```bash
conda create -n mms-code-switching python=3.10.2
```
```bash
pip install -r requirements.txt
```
This also installs editable modified code for transformers from this repository.

#### Download model checkpoints:

| Model                | ASCEND (MER / CER) | ESCWA (WER / CER) | MUCS (WER / CER) | 
|----------------------|--------------------|--------------------|-------------------|
| **MMS with single language adapter:** |           |            |                  |               
| English              | 98.02 / 87.85   | 92.73 / 71.14    | 101.72 / 74.02 |  
| Matrix-language      | 71.98 / 66.76   | 75.98 / 46.38    | 58.05 / 49.20  |  
| **Proposed models for fine-tuning:** |           |            |                  |               
| Matrix-language-FT   | 45.97 / 44.13   [Download](#)   | 77.47 / 37.69   [Download](#)    | 66.19 / 41.10  [Download](#)   | 
| Post Adapter Code Switching                 | 44.41 / 40.24   [Download](#)   | 75.50 / 46.69   [Download](#)    | 63.32 / 42.66   [Download](https://drive.google.com/file/d/1TjuIyugkKlW9_GiJU9vBV2SuLb-pRWfL/view?usp=drive_link)  | 
| Transformer Code Switching                  | 41.07 / 37.89   [Download](https://drive.google.com/file/d/1LzKnsYXvE1vImZj7TWkTGAxKJqBnMPN1/view?usp=drive_link)   | 74.42 / 35.54   [Download](https://drive.google.com/file/d/1hE9Cy3qo5XbEE3p1Lr1i3sTgfD6muGKp/view?usp=drive_link)    | 57.95 / 38.26  [Download](https://drive.google.com/file/d/1LzKnsYXvE1vImZj7TWkTGAxKJqBnMPN1/view?usp=drive_link)   | 


#### Do inference

Run below script to generate transcripts

Use `main` for Transformer Based Switching and `post-adapter-switching` for Post Adapter Swtiching. Ex:-

```bash
git checkout main
```
```bash
python inference.py --test_metadata_csv_path "/l/users/speech_lab/CodeSwitchedDataset[code_switched_dataset]/ASCEND/test_metadata.csv" --target_lang_1 eng --target_lang_2 cmn-script_simplified --prefix_path "/l/users/speech_lab/CodeSwitchedDataset[code_switched_dataset]/ASCEND/" --checkpoint_path "/l/users/speech_lab/AtharvaK/MMS-Adpter-Switching/checkpoints/mms_out_transformer_code_switcher/checkpoint-33200"  --batch_size 32
```

#### Output transcripts

We also share transcripts generated by our proposed systems on the 3 datasets in `transcripts/`. 
