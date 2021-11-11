import os
import torch
import numpy
import json
import tqdm
from transformers import BartTokenizer, BartForCausalLM
from Loader import loader_cnndm
from Tools import ProgressBar, save_model

save_path = 'E:/ProjectData/GSMLM-Bart-Base/'
if not os.path.exists(save_path): os.makedirs(save_path)


def only_context(batch_article):
    article_ids, article_label = batch_article['input_ids'].numpy(), batch_article['mlm_label'].numpy()

    treated_article_ids, treated_article_label = [], []
    for indexX in range(len(article_ids)):
        start_flag = False
        current_article_ids, current_article_label = [], []
        for indexY in range(1, len(article_ids[indexX])):
            if article_ids[indexX][indexY] == 0: start_flag = True
            if article_ids[indexX][indexY] == 1: continue
            if start_flag:
                current_article_ids.append(article_ids[indexX][indexY])
                current_article_label.append(article_label[indexX][indexY])

        treated_article_ids.append(current_article_ids)
        treated_article_label.append(current_article_label)

    max_len = numpy.max([len(_) for _ in treated_article_ids])
    for index in range(len(treated_article_ids)):
        treated_article_ids[index] = numpy.concatenate(
            [treated_article_ids[index], numpy.zeros(max_len - len(treated_article_ids[index]))])
        treated_article_label[index] = numpy.concatenate(
            [treated_article_label[index], [-100 for _ in range(max_len - len(treated_article_label[index]))]])
    return torch.LongTensor(treated_article_ids).cuda(), torch.LongTensor(treated_article_label)


def pad_context(batch_article):
    article_ids, article_label = batch_article['input_ids'].numpy(), batch_article['mlm_label'].numpy()

    treated_article_ids, treated_article_label = [], []
    for indexX in range(len(article_ids)):
        start_flag = False
        current_article_ids, current_article_label = [0], [-100]
        for indexY in range(1, len(article_ids[indexX])):
            if article_ids[indexX][indexY] == 0: start_flag = True
            if start_flag:
                current_article_ids.append(article_ids[indexX][indexY])
                current_article_label.append(article_label[indexX][indexY])
            else:
                current_article_ids.append(1)
                current_article_label.append(article_label[indexX][indexY])
        treated_article_ids.append(current_article_ids)
        treated_article_label.append(current_article_label)
    return torch.LongTensor(treated_article_ids).cuda(), torch.LongTensor(treated_article_label)


def predict_summary(batch_article):
    article_ids, article_label = batch_article['input_ids'].numpy(), batch_article['mlm_label'].numpy()

    treated_article_ids, treated_article_label = [], []
    for indexX in range(len(article_ids)):
        start_flag = False
        current_article_ids = predict_encode_result[batch_size * batch_index + indexX]
        current_article_label = [-100 for _ in range(len(current_article_ids))]

        for indexY in range(1, len(article_ids[indexX])):
            if article_ids[indexX][indexY] == 1: continue
            if article_ids[indexX][indexY] == 0: start_flag = True
            if start_flag:
                current_article_ids.append(article_ids[indexX][indexY])
                current_article_label.append(article_label[indexX][indexY])

        treated_article_ids.append(current_article_ids)
        treated_article_label.append(current_article_label)

    max_len = numpy.max([len(_) for _ in treated_article_ids])
    for index in range(len(treated_article_ids)):
        treated_article_ids[index] = numpy.concatenate(
            [treated_article_ids[index], numpy.zeros(max_len - len(treated_article_ids[index]))])
        treated_article_label[index] = numpy.concatenate(
            [treated_article_label[index], [-100 for _ in range(max_len - len(treated_article_label[index]))]])
    return torch.LongTensor(treated_article_ids).cuda(), torch.LongTensor(treated_article_label)


if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-base')
    train_loader, val_loader, test_loader = loader_cnndm(
        batch_size=3, tokenizer=tokenizer, keywords_name='SalientWords')

    # batch_size = 3
    # load_path = 'C:/PythonProject/DataSource-CNNDM-BART-Predict/test-vanilla'
    # predict_encode_result = []
    # for index in range(11490):
    #     current_data = json.load(open(os.path.join(load_path, '%08d.json' % index), 'r'))
    #     predict_encode_result.append(tokenizer.encode(current_data['predict'])[0:-1])

    # batch_number = 53999
    for batch_number in range(72999, 0, -1000):
        model = BartForCausalLM.from_pretrained('E:/ProjectData/GSMLM-Bart-Base/%08d-Encoder' % batch_number)

        if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.cuda()

        total_result = []
        for batch_index, batch_article in enumerate(tqdm.tqdm(test_loader)):
            # if batch_index > 10: break
            article_ids, article_label = batch_article['input_ids'].cuda(), batch_article['mlm_label']
            # article_ids, article_label = only_context(batch_article)
            # article_ids, article_label = pad_context(batch_article)
            # article_ids, article_label = predict_summary(batch_article)
            # print(tokenizer.decode(article_ids[0]))
            # exit()
            result = model(input_ids=article_ids)

            article_label = article_label.numpy()

            for indexX in range(numpy.shape(article_label)[0]):
                current_sample = {'predict': [], 'label': []}
                for indexY in range(numpy.shape(article_label)[1]):
                    if article_label[indexX][indexY] == -100: continue
                    current_sample['predict'].append(torch.argmax(result['logits'][indexX][indexY]).cpu().numpy())
                    current_sample['label'].append(article_label[indexX][indexY])
                current_sample['predict'] = [int(_) for _ in numpy.array(current_sample['predict'])]
                current_sample['label'] = [int(_) for _ in numpy.array(current_sample['label'])]
                total_result.append(current_sample)
            # break
        json.dump(total_result, open(os.path.join(save_path, '%08d-Predict-Gold.json' % batch_number), 'w'))
