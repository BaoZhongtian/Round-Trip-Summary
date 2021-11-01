import os
import torch
import tqdm
import numpy
from Loader import loader_cnndm
from transformers import BertForMaskedLM
from Tools import ProgressBar, save_model

# 0.93093

if __name__ == '__main__':
    train_data, val_data, test_data = loader_cnndm(batch_size=15, collate_method='summary_mlm')
    model = BertForMaskedLM.from_pretrained('E:/ProjectData/GoldSummaryMLM/00031999-Encoder')
    model.cuda()

    if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    total_predict, total_label = [], []

    for batch_index, batch_article in enumerate(tqdm.tqdm(test_data)):
        article_ids, article_mask = batch_article['input_ids'].cuda(), batch_article['attention_mask'].cuda()
        article_label = batch_article['mlm_label']
        batch_predict = model(input_ids=article_ids, attention_mask=article_mask)
        # print(batch_predict)
        # print(numpy.shape(batch_predict[0]))
        # exit()
        batch_predict = batch_predict[0].detach().cpu().numpy()

        for indexX in range(numpy.shape(batch_predict)[0]):
            for indexY in range(numpy.shape(batch_predict)[1]):
                if article_label[indexX][indexY] != -100:
                    total_predict.append(numpy.argmax(batch_predict[indexX][indexY]))
                    total_label.append(article_label[indexX][indexY].item())

    correct_number = numpy.sum([total_label[_] == total_predict[_] for _ in range(len(total_label))])
    print(float(correct_number) / len(total_label))
