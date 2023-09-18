"""
srun --ntasks=1 --cpus-per-task=4 -p gpu -q gpu-8 --mem=20G --gres=gpu:1 python train_mms.py --train_metadata_csv_path "/l/users/speech_lab/CodeSwitchedDataset[code_switched_dataset]/ASCEND/train_metadata.csv" --test_metadata_csv_path "/l/users/speech_lab/CodeSwitchedDataset[code_switched_dataset]/ASCEND/test_metadata.csv" --target_lang_1 eng --target_lang_2 cmn-script_simplified
"""

print('Loading Dependancies..')
import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTCWithAdapterSwitching
from datasets import load_dataset, load_metric, Audio
from datasets import Dataset
from transformers import  AutoProcessor, TrainingArguments, Trainer
import pandas as pd
from evaluate import load
from dataclasses import dataclass, field  
from typing import Any, Dict, List, Optional, Union
import argparse
import numpy as np
import torch
from safetensors.torch import load_file
import random

torch.manual_seed(69)
np.random.seed(69)
random.seed(69)

import jieba
import editdistance
from itertools import chain
from jiwer import wer, cer
from evaluate import load
import time

def check_adapters_loaded_correctly(model, lang1, lang2):
    adapter1 = load_file(f'/home/atharva.kulkarni/.cache/huggingface/hub/models--facebook--mms-1b-all/snapshots/3d33597edbdaaba14a8e858e2c8caa76e3cec0cd/adapter.{lang1}.safetensors')
    adapter2 = load_file(f'/home/atharva.kulkarni/.cache/huggingface/hub/models--facebook--mms-1b-all/snapshots/3d33597edbdaaba14a8e858e2c8caa76e3cec0cd/adapter.{lang2}.safetensors')
    
    for k,v in model.state_dict().items():
        if 'adapter_layer_1' in k:
            t1 = v
            t2 = adapter1[k.replace('adapter_layer_1', 'adapter_layer')]
            assert torch.equal(t1, t2)
        if 'adapter_layer_2' in k:
            t1 = v
            t2 = adapter2[k.replace('adapter_layer_2', 'adapter_layer')]
            assert torch.equal(t1, t2)


def tokenize_for_mer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, jieba.lcut(text)))
    tokens = [[tok] if tok.isascii() else list(tok) for tok in tokens]
    return list(chain(*tokens))

def tokenize_for_cer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, list(text)))
    return tokens


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs = processor.batch_decode(pred_ids)

    # we do not want to group tokens when computing the metrics
    label_strs = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer_eval = wer_metric.compute(predictions=pred_strs, references=label_strs)
    cer_eval = cer_metric.compute(predictions=pred_strs, references=label_strs)

    mixed_distance, mixed_tokens = 0, 0
    char_distance, char_tokens = 0, 0
    for pred_str, label_str in zip(pred_strs, label_strs):
        # Calculate 
        m_pred = tokenize_for_mer(pred_str)
        m_ref = tokenize_for_mer(label_str)
        mixed_distance += editdistance.distance(m_pred, m_ref)
        mixed_tokens += len(m_ref)

        c_pred = tokenize_for_cer(pred_str)
        c_ref = tokenize_for_cer(label_str)
        char_distance += editdistance.distance(c_pred, c_ref)
        char_tokens += len(c_ref)
    
    mer = mixed_distance / mixed_tokens
    cer = char_distance / char_tokens
    print({"mer": mer, "cer": cer, "wer_eval": wer_eval, "cer_eval" : cer_eval})
    return {"mer": mer, "cer": cer, "wer_eval": wer_eval, "cer_eval" : cer_eval}

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    batch["labels"] = processor(text=batch["transcripts"]).input_ids
    return batch


def main():
    global wer_metric
    global cer_metric 

    wer_metric = load('wer')
    cer_metric = load('cer')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_metadata_csv_path', default=None)
    parser.add_argument('--target_lang_1', required=True)
    parser.add_argument('--target_lang_2', required=True)
    parser.add_argument('--trial_run')    
    parser.add_argument('--prefix_path', required=True)
    parser.add_argument('--checkpoint_path', required=False)
    parser.add_argument('--outfile_path', required=False)
    parser.add_argument('--batch_size', required=True, type=int)
    args = parser.parse_args()


    print('Loading Datasets...')

    try:
        if args.trial_run:
            df_test = pd.read_csv(args.test_metadata_csv_path, usecols=['file_name', 'transcription'], sep="|").head(args.batch_size)
        else:
            df_test = pd.read_csv(args.test_metadata_csv_path, usecols=['file_name', 'transcription'], sep="|")
    except:
        if args.trial_run:
            df_test = pd.read_csv(args.test_metadata_csv_path, usecols=['file_name', 'transcription']).head(args.batch_size)
        else:
            df_test = pd.read_csv(args.test_metadata_csv_path, usecols=['file_name', 'transcription'])

    prefix_path = args.prefix_path

    df_test['file_name'] = df_test['file_name'].map(lambda x: prefix_path + x)
    
    test_data = Dataset.from_dict({'audio' : df_test['file_name'], 'transcripts':df_test['transcription']}).cast_column("audio", Audio())
    
    print('Loading Model and Processor..')
    global processor
    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    processor.tokenizer.set_code_switched_target_langs(args.target_lang_1, args.target_lang_2)
    
    test_data = test_data.map(prepare_dataset, remove_columns=['audio', 'transcripts'], num_proc=8)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTCWithAdapterSwitching.from_pretrained(
        "facebook/mms-1b-all",
        config = 'post_adapter',
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

    model.load_adapters_for_code_switching(args.target_lang_1, args.target_lang_2)
    model.load_state_dict(torch.load(args.checkpoint_path + '/pytorch_model.bin'))

    print(model)

    check_adapters_loaded_correctly(model, args.target_lang_1, args.target_lang_2)
    print('Checked adapter loading')
            
    print('Setting up training..')
    training_args = TrainingArguments(
        output_dir = 'test/',
        group_by_length=True,
        length_column_name = 'input_length',
        per_device_eval_batch_size = args.batch_size,
        report_to="none",
        eval_accumulation_steps = 10

    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        eval_dataset=test_data,
        tokenizer=processor,
        compute_metrics = compute_metrics
    )

    print('Staring evaluation..')
    
    results = trainer.predict(test_data)

    ids = torch.argmax(torch.tensor(results[0]), dim=-1)
    transcription = processor.batch_decode(ids)
    df = df_test.copy()
    df['mms_model_transcription'] = transcription
    df.to_csv(args.outfile_path)
    
if __name__ == '__main__':

    main()
