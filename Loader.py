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


def loader_cnndm(batch_size=4, collate_method='Basic'):
    def collate_basic(input_data):
        batch_article, batch_summary = [_['article'] for _ in input_data], [_['summary'] for _ in input_data]
        batch_article_token = tokenizer.batch_encode_plus(
            batch_article, max_length=512, return_tensors='pt', pad_to_max_length=True)
        batch_summary_token = tokenizer.batch_encode_plus(
            batch_summary, max_length=128, return_tensors='pt', pad_to_max_length=True)
        return batch_article_token, batch_summary_token

    def collate_summary_mlm(input_data):
        batch_article, batch_summary = [_['article'] for _ in input_data], [_['summary'] for _ in input_data]
        batch_word = [_['words'] for _ in input_data]

        encode_result = {'input_ids': [], 'attention_mask': [], 'mlm_label': []}
        for indexX in range(len(batch_article)):
            sample_article_token = tokenizer.encode(
                batch_article[indexX], add_special_tokens=False, max_length=384)[0:384]
            sample_summary_token = tokenizer.encode(
                batch_summary[indexX], add_special_tokens=False, max_length=128)[0:128]

            sample_salient_words = [_[0] for _ in batch_word[indexX]]
            sample_salient_tokens = [tokenizer.encode(_, add_special_tokens=False) for _ in sample_salient_words]

            indexY = 0
            masked_article_token, masked_article_label = [], []
            while indexY < len(sample_article_token):
                same_flag = False
                for check_words in sample_salient_tokens:
                    if same_flag: break
                    compare_words = sample_article_token[indexY:indexY + len(check_words)]
                    if len(compare_words) != len(check_words): continue
                    distance = numpy.sum(numpy.abs(numpy.array(compare_words) - numpy.array(check_words)))
                    if distance == 0:
                        same_flag = True
                        indexY += len(compare_words) - 1
                        masked_article_token.extend([103 for _ in range(len(compare_words))])
                        masked_article_label.extend(compare_words)
                        continue
                if not same_flag:
                    masked_article_token.append(sample_article_token[indexY])
                    masked_article_label.append(-100)
                indexY += 1

            token_ids = numpy.concatenate([[101], sample_summary_token, [102], masked_article_token])[0:511].tolist()
            token_ids.append(102)
            masked_article_label = numpy.concatenate(
                [[-100 for _ in range(len(sample_summary_token) + 2)], masked_article_label])
            if len(masked_article_label) > len(token_ids): masked_article_label = masked_article_label[0:len(token_ids)]
            if len(masked_article_label) < len(token_ids): masked_article_label = numpy.concatenate(
                [masked_article_label, [-100 for _ in range(len(token_ids) - len(masked_article_label))]])

            encode_result['input_ids'].append(numpy.concatenate([token_ids, numpy.zeros(512 - len(token_ids))]))
            encode_result['attention_mask'].append(
                numpy.concatenate([numpy.ones(len(token_ids)), numpy.zeros(512 - len(token_ids))]))
            encode_result['mlm_label'].append(
                numpy.concatenate([masked_article_label, [-100 for _ in range(512 - len(masked_article_label))]]))

        encode_result['input_ids'] = torch.LongTensor(encode_result['input_ids'])
        encode_result['attention_mask'] = torch.LongTensor(encode_result['attention_mask'])
        encode_result['mlm_label'] = torch.LongTensor(encode_result['mlm_label'])
        return encode_result

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    load_path = 'C:/PythonProject/DataSource/'
    train_data = json.load(open(os.path.join(load_path, 'CNNDM_train.json'), 'r'))
    val_data = json.load(open(os.path.join(load_path, 'CNNDM_val.json'), 'r'))
    test_data = json.load(open(os.path.join(load_path, 'CNNDM_test.json'), 'r'))
    print('Load Completed')

    collate_function = collate_basic
    if collate_method == 'summary_mlm':
        collate_function = collate_summary_mlm
        train_salient_words = json.load(open(os.path.join(load_path, 'CNNDM_SalientWords_train.json'), 'r'))
        val_salient_words = json.load(open(os.path.join(load_path, 'CNNDM_SalientWords_val.json'), 'r'))
        test_salient_words = json.load(open(os.path.join(load_path, 'CNNDM_SalientWords_test.json'), 'r'))

        for index in range(len(train_data)):
            assert train_data[index]['filename'] == train_salient_words[index]['filename']
            train_data[index]['words'] = train_salient_words[index]['words']
        for index in range(len(val_data)):
            assert val_data[index]['filename'] == val_salient_words[index]['filename']
            val_data[index]['words'] = val_salient_words[index]['words']
        for index in range(len(test_data)):
            assert test_data[index]['filename'] == test_salient_words[index]['filename']
            test_data[index]['words'] = test_salient_words[index]['words']

    # train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_function)
    val_dataset = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_function)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_function)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    train_loader, val_loader, test_loader = loader_cnndm(collate_method='summary_mlm')
    for batch_index, batch_article in enumerate(train_loader):
        # print(batch_article['attention_mask'])
        # print('\n\n')
        # print(batch_article['mlm_label'])
        # exit()
        print(batch_index, numpy.shape(batch_article['input_ids']), numpy.shape(batch_article['attention_mask']),
              numpy.shape(batch_article['mlm_label']))
        exit()
