import os
import tqdm
import numpy
import json
from Loader import loader_cnndm
from transformers import BartTokenizer

if __name__ == '__main__':
    batch_size = 3
    load_path = 'C:/PythonProject/DataSource-CNNDM-BART-Predict/test-vanilla'
    tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large')
    predict_encode_result = []
    for index in range(11490):
        current_data = json.load(open(os.path.join(load_path, '%08d.json' % index), 'r'))
        predict_encode_result.append(tokenizer.encode(current_data['predict'])[0:-1])

    train_loader, val_loader, test_loader = loader_cnndm(
        batch_size=3, tokenizer=tokenizer, keywords_name='SalientWords')
    for batch_index, batch_article in enumerate(tqdm.tqdm(test_loader)):
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
        print(numpy.shape(treated_article_ids), numpy.shape(treated_article_label))
        exit()
