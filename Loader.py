import os
import json
import numpy
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BartTokenizer

LOAD_PATH = 'C:/PythonProject/DataSource/'


# LOAD_PATH = 'D:/ProjectData/CNNDM/'


class CollateClass:
    def __init__(self, tokenizer, max_article_length=896, max_summary_length=128, keywords_name=None):
        self.tokenizer = tokenizer
        self.max_article_length, self.max_summary_length = max_article_length, max_summary_length
        self.keywords_dictionary = {}
        if keywords_name is not None:
            for part in ['train', 'val', 'test']:
                current_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_%s_%s.json' % (keywords_name, part))))
                for sample in current_data:
                    self.keywords_dictionary[sample['filename']] = sample['words']

    def collate(self, input_data):
        if len(self.keywords_dictionary.keys()) != 0:
            return self.collate_keywords(input_data)
        else:
            return self.collate_basic(input_data)

    def collate_basic(self, input_data):
        batch_article, batch_summary = [_['article'] for _ in input_data], [_['summary'] for _ in input_data]
        batch_article_token = self.tokenizer.batch_encode_plus(
            batch_article, max_length=self.max_article_length, padding=True, truncation=True, return_tensors='pt')
        batch_summary_token = self.tokenizer.batch_encode_plus(
            batch_summary, max_length=self.max_summary_length, padding=True, truncation=True, return_tensors='pt')
        return batch_article_token, batch_summary_token

    def collate_keywords(self, input_data):
        mask_id = self.tokenizer.convert_tokens_to_ids(['<mask>'])[0]
        pad_id = self.tokenizer.convert_tokens_to_ids(['<pad>'])[0]

        batch_token, batch_lm_label = [], []
        for index in range(len(input_data)):
            current_summary_token = self.tokenizer.encode_plus(
                input_data[index]['summary'], add_special_tokens=True, max_length=self.max_summary_length)['input_ids']

            current_lm_label = []
            # current_lm_label = [-100 for _ in range(len(current_token))]

            #######################################
            current_keywords = self.keywords_dictionary[input_data[index]['filename']]
            current_keywords_tokens = self.tokenizer.batch_encode_plus(
                [' ' + _[0] for _ in current_keywords], add_special_tokens=False)['input_ids']
            current_article_token = self.tokenizer.encode_plus(
                input_data[index]['article'], add_special_tokens=True, max_length=self.max_article_length,
                truncation=True)['input_ids']

            indexX = -1
            while indexX < len(current_article_token):
                indexX += 1
                similar_flag = False
                for indexY in range(len(current_keywords_tokens)):
                    for indexZ in range(len(current_keywords_tokens[indexY])):
                        if indexX + indexZ >= len(current_article_token): break
                        if current_article_token[indexX + indexZ] != current_keywords_tokens[indexY][indexZ]: break
                    if indexX + indexZ >= len(current_article_token): continue
                    if current_article_token[indexX + indexZ] == current_keywords_tokens[indexY][indexZ]:
                        similar_flag = True
                        break
                if similar_flag:
                    for indexZ in range(len(current_keywords_tokens[indexY])):
                        current_lm_label.append(current_article_token[indexX + indexZ])
                        current_article_token[indexX + indexZ] = mask_id
                    indexX += indexZ
                    continue
                current_lm_label.append(-100)

            current_lm_label = current_lm_label[0:len(current_article_token)]
            assert len(current_lm_label) == len(current_article_token)

            current_lm_label = numpy.concatenate(
                [[-100 for _ in range(len(current_summary_token) - 1)], current_lm_label])
            current_input_ids = numpy.concatenate([current_summary_token[0:-1], current_article_token])

            batch_token.append(current_input_ids)
            batch_lm_label.append(current_lm_label)

        treated_input_ids, treated_lm_label = [], []
        treated_length = max([len(_) for _ in batch_token])
        for index in range(len(batch_token)):
            treated_input_ids.append(numpy.concatenate(
                [batch_token[index], [pad_id for _ in range(treated_length - len(batch_token[index]))]]))

        treated_length = max([len(_) for _ in batch_lm_label])
        for index in range(len(batch_lm_label)):
            treated_lm_label.append(numpy.concatenate(
                [batch_lm_label[index], [-100 for _ in range(treated_length - len(batch_lm_label[index]))]]))
        return {'input_ids': torch.LongTensor(treated_input_ids), 'mlm_label': torch.LongTensor(treated_lm_label)}


def loader_cnndm(
        batch_size=4, tokenizer=None, train_part_shuffle=True, max_article_length=768, max_summary_length=256,
        limit_size=None, keywords_name=None):
    if tokenizer is None: tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_train.json'), 'r'))
    val_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_val.json'), 'r'))
    test_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_test.json'), 'r'))
    print('Load Completed')
    if limit_size is not None:
        train_data = train_data[0:limit_size]
        val_data = val_data[0:limit_size]
        test_data = test_data[0:limit_size]

    collate = CollateClass(tokenizer, max_article_length, max_summary_length, keywords_name=keywords_name)

    train_dataset = DataLoader(
        train_data, batch_size=batch_size, shuffle=train_part_shuffle, collate_fn=collate.collate)
    val_dataset = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, collate_fn=collate.collate)
    test_dataset = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate.collate)

    # with open('Result.csv', 'w') as file:
    #     for batch_data in val_dataset:
    #         for indexX in range(len(batch_data['input_ids'])):
    #             for indexY in range(len(batch_data['input_ids'][indexX])):
    #                 file.write(str(batch_data['input_ids'][indexX][indexY]) + ',')
    #             file.write('\n')
    #             for indexY in range(len(batch_data['lm_label'][indexX])):
    #                 file.write(str(batch_data['lm_label'][indexX][indexY]) + ',')
    #             file.write('\n')
    #         exit()
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large')
    train_loader, val_loader, test_loader = loader_cnndm(
        batch_size=4, tokenizer=tokenizer, keywords_name='SalientWords')
    for sample in val_loader:
        print(sample)
        exit()
#
#     article_len, summary_len = [], []
#     for batch_data in tqdm.tqdm(train_loader):
#         article_len.append(numpy.shape(batch_data[0]['input_ids'])[1])
#         summary_len.append(numpy.shape(batch_data[1]['input_ids'])[1])
#     for batch_data in tqdm.tqdm(val_loader):
#         article_len.append(numpy.shape(batch_data[0]['input_ids'])[1])
#         summary_len.append(numpy.shape(batch_data[1]['input_ids'])[1])
#     for batch_data in tqdm.tqdm(test_loader):
#         article_len.append(numpy.shape(batch_data[0]['input_ids'])[1])
#         summary_len.append(numpy.shape(batch_data[1]['input_ids'])[1])
#     print(article_len)
#     import json
#
#     json.dump(article_len, open('article_len.json', 'w'))
#     json.dump(summary_len, open('summary_len.json', 'w'))
