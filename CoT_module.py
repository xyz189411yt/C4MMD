"""
IMPORTANT!!

This module code mainly refers to the demo released by InternLM XCompositer around October 2023.
At present, the demo has been updated. If it cannot run normally or there are problems, you can visit the project according to the following link to find the currently available demo.
https://github.com/InternLM/InternLM-XComposer

The main code of the CoT module can refer to the content after line 137.
You can use the latest MLLM generation tool to simultaneously splice our CoT module.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import sys
import json
from PIL import ImageFile

import torch
from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
    Blip2ForConditionalGeneration,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    AutoTokenizer,
)

from arguments import ModelArguments, DataTrainingArguments

import auto_gptq
from auto_gptq.modeling import BaseGPTQForCausalLM

auto_gptq.modeling._base.SUPPORTED_MODELS = ["InternLMXComposer"]

class InternLMXComposerQForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "internlm_model.model.layers"
    outside_layer_modules = [
        "query_tokens",
        "flag_image_start",
        "flag_image_end",
        "visual_encoder",
        "Qformer",
        "internlm_model.model.embed_tokens",
        "internlm_model.model.norm",
        "internlm_proj",
        "internlm_model.lm_head",
    ]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj"],
        ["mlp.up_proj"],
        ["mlp.down_proj"],
    ]


logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.set_grad_enabled(False)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed)

# Load dataset
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
    extension = data_args.validation_file.split(".")[-1]
if data_args.test_file is not None:
    data_files["test"] = data_args.test_file
    extension = data_args.test_file.split(".")[-1]


config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
config.pre_seq_len = model_args.pre_seq_len
config.prefix_projection = model_args.prefix_projection

device = 'cuda:0'

model = InternLMXComposerQForCausalLM.from_quantized(
            model_args.model_name_or_path, trust_remote_code=True, device="cuda:0"
        )
tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
model.model.tokenizer = tokenizer
model.config.max_length = 100

max_target_length = data_args.max_target_length


def save_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# =======================CoT module=======================

process_file_list = ['train_data', 'val_data', 'test_data']
origin_data_path = '''
The origin data path, the one you saved from https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors
'''
new_data_path = '''
The data path you processed with MLLM.
'''
image_file_path = '''
Fill in the image save path here, which should have two folders containing Chinese and English images, respectively.
For example:

data/image -> the image file path, which contains two folders as follow.
    |_English
    |_Chinese
'''

for file in process_file_list:
    # Check first if there are any checkpoints present
    if os.path.exists(f'{new_data_path}/{file}.json'):
        datas = load_json(f'{new_data_path}/new_{file}.json')
    else:
        datas = load_json(f'{origin_data_path}/new_{file}.json')
    count = 0
    for line in tqdm(datas):
        if 'internlm_mix_info' in line:
            continue
        img_name = line['file_name']
        img = f'{image_file_path}/{img_name}'
        Question1 = 'Please temporarily ignore the text in the image and describe the content in the image. Try to be concise while ensuring the correctness of your answers.'
        inputs = {'text': Question1, 'image': img}
        response1 = model.generate(**inputs)
        line['internlm_img_info'] = response1

        Question2 = f'The text in the picture is as follows: "{line["text"]}". Please analyze the meaning of the text. Note that there may be homophonic memes and puns, distinguish and explain them but do not over interpret while ensuring the correctness of the answer and be concise.'
        inputs = {'text': Question2}
        response2 = model.generate(**inputs)
        line['internlm_text_info'] = response2

        Question3 = f'Image description: {response1}; Text: "{line["text"]}"; Text description: {response2}. Please combine the image, text, and their description information and try to understand the deep meaning of the combination of the image and text. No need to describe images and text, only answer implicit meanings. Ensure the accuracy of the answer and try to be concise as much as possible.'
        inputs = {'text': Question3, 'image': img}
        response3 = model.generate(**inputs)
        line['internlm_mix_info'] = response3

        count += 1
        if count == 100:
            # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
            print('save_a_part')
            count = 0
            save_json(f'{new_data_path}/new_{file}.json', datas)

    save_json(f'{new_data_path}/new_{file}.json', datas)

print('finish!')