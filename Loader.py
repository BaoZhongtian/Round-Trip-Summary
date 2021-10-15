import os
import json
import tqdm
import torch
import numpy
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


# class DatasetForAll(Dataset):
#     def __init__(self, input_data):
#         self.data = input_data
#
#     def __getitem__(self, item):
#         return self.data[item]
#
#     def __len__(self):
#         return len(self.data)


def loader_cnndm(batch_size=4):
    def collate(input_data):
        batch_article, batch_summary = [_['article'] for _ in input_data], [_['summary'] for _ in input_data]
        batch_article_token = tokenizer.batch_encode_plus(
            batch_article, max_length=512, return_tensors='pt', pad_to_max_length=True)
        batch_summary_token = tokenizer.batch_encode_plus(
            batch_summary, max_length=128, return_tensors='pt', pad_to_max_length=True)
        return batch_article_token, batch_summary_token

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    load_path = 'C:/PythonProject/DataSource/'
    train_data = json.load(open(os.path.join(load_path, 'CNNDM_train.json'), 'r'))
    val_data = json.load(open(os.path.join(load_path, 'CNNDM_val.json'), 'r'))
    test_data = json.load(open(os.path.join(load_path, 'CNNDM_test.json'), 'r'))
    print('Load Completed')

    # train_dataset = DatasetForAll(train_data)
    # val_dataset = DatasetForAll(val_data)
    # test_dataset = DatasetForAll(test_data)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dataset = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    train_loader, val_loader, test_loader = loader_cnndm()
    for batch_index, [batch_article, batch_summary] in enumerate(train_loader):
        # print(batch_article)
        print(batch_index, numpy.shape(batch_article['input_ids']), numpy.shape(batch_summary['input_ids']))
        # exit()
