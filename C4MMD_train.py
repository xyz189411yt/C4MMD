"""
IMPORTANT!!

The main training process of C4MMD.
You can find the model structure of C4MMD, data processing from ./data/ and the process of training, and all parameter settings.

We did not set the parameters in a form that can be set on the command line. Because that would increase the amount of code.
We hope to help you reproduce our work with minimal code.
"""


import logging
import torch
from torch import nn as nn
import json
from sklearn import metrics
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import ViTFeatureExtractor, ViTModel, XLMRobertaModel


import time
import random
import numpy as np
import os
import datetime
from tqdm import tqdm


trained_model_path = f'trained_models/C4MMD.pth'
image_file_path = '''
Fill in the image save path here, which should have two folders containing Chinese and English images, respectively.
For example:

data/image -> the image file path, which contains two folders as follow.
    |_English
    |_Chinese
'''
log_file_name = 'C4MMD'

language_model = '''
Your saved language model path, for example:

xlm-roberta-base

If you want to reproduce our results, please use the example model type above.
'''
vision_model = '''
Your saved vision model path, for example:

vit-base-patch16-224

If you want to reproduce our results, please use the example model type above.
'''

train_data = '''
your training data, you can get data type in the data folder. For example:
data/train_data.json
'''
val_data = '''
your training data, you can get data type in the data folder. For example:
data/val_data.json
'''
test_data = '''
your training data, you can get data type in the data folder. For example:
data/test_data.json
'''

do_train = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ImageFile.LOAD_TRUNCATED_IMAGES = True
NUM_LABELS = 2
LR = 1e-5
EPOCHES = 100
MAX_LEN = 32
seed = 42
BATCH_SIZE = 8

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class VitBert(nn.Module):
    def __init__(self, vit, bert, num_labels):
        super(VitBert, self).__init__()
        self.vit = vit
        self.bert = bert
        self.num_labels = num_labels
        config = vit.config
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.text_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.img_classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.bert.config.type_vocab_size = 4
        self.bert.embeddings.token_type_embeddings = nn.Embedding(
            self.bert.config.type_vocab_size, config.hidden_size
        )
        self.bert._init_weights(bert.embeddings.token_type_embeddings)

    def forward(self, text, text_attention, vilt_img, token_type_ids):
        text_hidden_states = self.bert(input_ids=text, attention_mask=text_attention, token_type_ids=token_type_ids)[0]
        outputs = self.vit(pixel_values=vilt_img,)
        hidden_states, pool_output = outputs[0], outputs[1]

        img_features = torch.sum(text_hidden_states * (token_type_ids == 1)[:, :, None], dim=1) / torch.sum(token_type_ids == 1, dim=1)[:, None]
        img_features = torch.cat([pool_output, img_features], dim=-1)

        text_features = torch.sum(text_hidden_states * (token_type_ids == 2)[:, :, None], dim=1) / torch.sum(token_type_ids == 2, dim=1)[:, None]

        hidden_states_mean = torch.sum(text_hidden_states * text_attention[:, :, None], dim=1) / torch.sum(text_attention, dim=1)[:, None]
        mix_fitures = torch.cat([pool_output, hidden_states_mean], dim=-1)

        logits = self.classifier(mix_fitures)
        img_logits = self.img_classifier(img_features)
        text_logits = self.text_classifier(text_features)
        return logits, img_logits, text_logits


class Collator(object):
    def __init__(self, tokenizer, processor):
        self.tokenizer = tokenizer
        self.processor = processor

    def __call__(self, batch):
        max_text_length = max([len(line[0]) for line in batch])
        input_ids, attention_mask, token_type_ids = [], [], []
        for line in batch:
            inputs, attention = line[0], line[1]
            token_type_id = line[6]
            input_ids.append(inputs + [self.tokenizer.pad_token_id] * (max_text_length - len(inputs)))
            attention_mask.append(attention + [0] * (max_text_length - len(attention)))
            token_type_ids.append(token_type_id + [0] * (max_text_length - len(token_type_id)))
        pixel_value = torch.stack([line[2] for line in batch]).squeeze(1)
        lables1, text_lables, img_lables = torch.tensor([line[3] for line in batch]), torch.tensor([line[4] for line in batch]), torch.tensor([line[5] for line in batch])

        outputs = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'pixel_value': pixel_value,
            'lables1': lables1,
            'text_labels': text_lables,
            'img_labels': img_lables,
            'token_type_ids': torch.tensor(token_type_ids),
        }
        return outputs


class VitXLMRDataset(torch.utils.data.Dataset):
    def __init__(self, path, processor, tokenizer, usage="train"):
        self.path = path
        self.datas = load_json(path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.usage = usage

    def __len__(self):
        return len(self.datas)

    def convert_str_to_ids(self, discription, token_type_id, head_space=True, max_id_num=100):
        inputs = []
        for i, token in enumerate(discription.split()):
            token = token if i == 0 and not head_space else ' ' + token
            tokenized_token = self.tokenizer(token, add_special_tokens=False)
            inputs += tokenized_token['input_ids']

        inputs = inputs[: max_id_num] if len(inputs) > max_id_num else inputs
        type_ids = [token_type_id] * len(inputs)
        return inputs, type_ids

    def __getitem__(self, idx):
        line = self.datas[idx]
        img, text, lable, lable2 = line['file_name'], line['text'], int(line['metaphor occurrence']), line['metaphor category']
        img_info = line['internlm_img_info']
        text_info = line['internlm_text_info']
        mix_info = line['internlm_mix_info']
        img_info = 'None.' if img_info.strip() == '' else img_info
        text_info = text + " " + text_info if text_info.strip() != '' else text
        text_info = 'None.' if img_info.strip() == '' else text_info
        img = Image.open(f'{image_file_path}/{img}')
        if img.mode != 'RGB':
            img = img.convert("RGB")
        img_encoding = self.processor(img, padding="max_length", truncation=True, return_tensors='pt')
        img.close()

        image_inputs, image_type_ids = self.convert_str_to_ids(img_info, 1, head_space=False, max_id_num=100)
        text_inputs, text_type_ids = self.convert_str_to_ids(text_info, 2, max_id_num=100)
        mix_inputs, mix_type_ids = self.convert_str_to_ids(mix_info, 3, max_id_num=100)


        discription_inputs = [self.tokenizer.cls_token_id] + image_inputs + text_inputs + mix_inputs + [self.tokenizer.sep_token_id]
        discription_attention = [1] * len(discription_inputs)
        discription_type_ids = [0] + image_type_ids + text_type_ids + mix_type_ids + [0]

        if 1 not in discription_type_ids or 2 not in discription_type_ids:
            print(discription_type_ids)

        if lable2 == '' or 'complementary' in lable2:
            img_lable = 0
            text_lable = 0
        elif 'text' in lable2:
            img_lable = 0
            text_lable = 1
        elif 'image' in lable2:
            img_lable = 1
            text_lable = 0
        else:
            raise KeyError

        return discription_inputs, discription_attention, img_encoding['pixel_values'], lable, text_lable, img_lable, discription_type_ids


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def num_correct(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return np.sum(preds == labels)


def save_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_best_ACC(f1_log):
    if not os.path.exists('acc_log'):
        os.makedirs('acc_log')

    with open(f'acc_log/{log_file_name}.txt', 'a' if os.path.exists(f'acc_log/{log_file_name}.txt') else 'w') as f:
        json.dump(f1_log, f)
        f.write('\n')


def save_best_F1(f1_log):
    if not os.path.exists('f1_log'):
        os.makedirs('f1_log')

    with open(f'f1_log/{log_file_name}.txt', 'a' if os.path.exists(f'f1_log/{log_file_name}.txt') else 'w') as f:
        json.dump(f1_log, f)
        f.write('\n')


if not os.path.exists('log'):
    os.makedirs('log')
file_handler = logging.FileHandler(f'log/{log_file_name}.txt')
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(f"device:{device}")

tokenizer = AutoTokenizer.from_pretrained(language_model)
processor = ViTFeatureExtractor.from_pretrained(vision_model)


def train():
    bert = XLMRobertaModel.from_pretrained(language_model)
    vit = ViTModel.from_pretrained(vision_model)
    model = VitBert(vit, bert, 2)
    train_dataset = VitXLMRDataset(train_data, processor, tokenizer, usage="train")
    val_dataset = VitXLMRDataset(val_data, processor, tokenizer, usage="val")
    test_dataset = VitXLMRDataset(test_data, processor, tokenizer, usage="test")

    logger.info('{:>5,} training samples'.format(len(train_dataset)))
    logger.info('{:>5,} validation samples'.format(len(val_dataset)))
    logger.info('{:>5,} test samples'.format(len(test_dataset)))
    best_val_f1_score = {'f1': -1,
                         'recall': -1,
                         'precision': -1}

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=BATCH_SIZE,
        collate_fn=Collator(tokenizer, processor)
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=BATCH_SIZE,
        collate_fn=Collator(tokenizer, processor)
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=BATCH_SIZE,
        collate_fn=Collator(tokenizer, processor)
    )

    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=LR,
                      eps=1e-8
                      )

    total_steps = len(train_dataloader) * EPOCHES

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    loss_func = torch.nn.CrossEntropyLoss()

    total_t0 = time.time()
    best_count = 0
    if do_train:
        for epoch_i in range(0, EPOCHES):
            logger.info("")
            logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHES))
            logger.info('Training...')

            t0 = time.time()

            total_train_loss = 0

            model.train()

            for batch in tqdm(train_dataloader, desc="Training"):
                text = batch['input_ids'].to(device)
                text_attention = batch['attention_mask'].to(device)
                image = batch['pixel_value'].to(device)
                labels = batch['lables1'].to(device)

                text_lables = batch['text_labels'].to(device)
                img_lables = batch['img_labels'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)

                model.zero_grad()

                logits, img_logits, text_logits = model(text, text_attention, image, token_type_ids)

                mix_loss = loss_func(logits, labels.long())
                text_loss = loss_func(text_logits, text_lables.long())
                img_loss = loss_func(img_logits, img_lables.long())

                total_loss = mix_loss + 0.5 * text_loss + 0.5 * img_loss

                total_train_loss += total_loss.item()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)

            training_time = format_time(time.time() - t0)

            logger.info("")
            logger.info("  Average training loss1: {0:.2f}".format(avg_train_loss))
            logger.info("  Training epcoh took: {:}".format(training_time))

            val_f1_score = evaluation(model, validation_dataloader, epoch_i)
            if best_val_f1_score['f1'] < val_f1_score['f1']:
                best_val_f1_score = val_f1_score
                torch.save(model.state_dict(), trained_model_path)
                best_count = 0
            else:
                best_count += 1
                if best_count > 10:
                    logger.info("Early Stop!")
                    break

    model.load_state_dict(torch.load(trained_model_path))
    best_test_f1_score = evaluation(model, test_dataloader, 0, val=False, save=True)

    f1_log = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'Epoch': EPOCHES,
        'LR': LR,
        'MAX_LEN': MAX_LEN,
        'F1_best_val': best_val_f1_score,
        'F1_best_test': best_test_f1_score,
    }
    save_best_F1(f1_log)
    logger.info("")
    logger.info("Training complete!")
    for key in f1_log:
        logger.info(f'{key}: {f1_log[key]}')
    logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    return f1_log


def evaluation(model, dataloader, epoch, val=True, save=False):
    logger.info("")
    logger.info("Running Validation..." if val else "Running Test...")

    t0 = time.time()

    model.eval()

    total_pred = np.array([], dtype=int)
    total_gold = np.array([], dtype=int)

    for batch in tqdm(dataloader, desc="Evaluating" if val else "Testing"):
        text = batch['input_ids'].to(device)
        text_attention = batch['attention_mask'].to(device)
        image = batch['pixel_value'].to(device)
        labels = batch['lables1'].to(device)

        token_type_ids = batch['token_type_ids'].to(device)

        with torch.no_grad():
            logits, img_logits, text_logits = model(text, text_attention, image, token_type_ids)
            predict = torch.argmax(logits.cpu().detach(), dim=-1)

            total_gold = np.append(total_gold, labels.cpu())
            total_pred = np.append(total_pred, predict)


    acc = round(metrics.accuracy_score(total_gold, total_pred) * 100, 2)
    report = metrics.classification_report(total_gold, total_pred, target_names=['0', '1'], digits=4)
    confusion = metrics.confusion_matrix(total_gold, total_pred).tolist()
    f1 = round(metrics.f1_score(total_gold, total_pred) * 100, 2)
    recall = round(metrics.recall_score(total_gold, total_pred) * 100, 2)
    precision = round(metrics.precision_score(total_gold, total_pred) * 100, 2)
    record = {
        'epoch': epoch + 1,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'acc': acc,
        'report': report,
        'confusion': confusion,

    }
    logger.info("")
    for key in sorted(record.keys()):
        logger.info(f"  {key} = {str(record[key])}")
    logger.info("")

    if save:
        saved_list = [[int(i), int(j)] for i, j in zip(total_pred, total_gold)]
        save_json('VitXLMR_argument2_test_result.json', saved_list)

    validation_time = format_time(time.time() - t0)

    logger.info("  Validation took: {:}".format(validation_time))

    return record


f1_list, recall_list, precision_list, acc_list = [], [], [], []
times = 5

for _ in range(times):
    f1_log = train()
    f1_list.append(f1_log['F1_best_test']['f1'])
    recall_list.append(f1_log['F1_best_test']['recall'])
    precision_list.append(f1_log['F1_best_test']['precision'])
    acc_list.append(f1_log['F1_best_test']['acc'])

logger.info(f'Average F1/Recall/Precision/ACC: {round(sum(f1_list) / times, 2)}/{round(sum(recall_list) / times, 2)}/{round(sum(precision_list) / times, 2)}/{round(sum(acc_list) / times, 2)}')

logger.info(f'end')

