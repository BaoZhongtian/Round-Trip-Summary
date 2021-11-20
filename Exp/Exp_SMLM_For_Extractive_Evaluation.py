import os
import json
import tqdm
import numpy
from rouge_score import rouge_scorer

if __name__ == '__main__':
    load_path = 'E:/ProjectData/CNNDM_SMLM_Result'
    cnn_path = 'C:/ProjectData/CNNDM_Dataset/cnn_stories/cnn/stories/'
    dm_path = 'C:/ProjectData/CNNDM_Dataset/dailymail_stories/dailymail/stories'
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    test_data = json.load(open('C:/PythonProject/DataSource/CNNDM_test.json', 'r'))
    test_filename = [_['filename'] + '.story' for _ in test_data]

    total_score_min, total_score_max = [], []
    for sample_filename in tqdm.tqdm(test_filename):
        sample_article, sample_summary = [], ''
        if os.path.exists(os.path.join(cnn_path, sample_filename)):
            with open(os.path.join(cnn_path, sample_filename), 'r', encoding='UTF-8') as file:
                data = file.readlines()
        else:
            with open(os.path.join(dm_path, sample_filename), 'r', encoding='UTF-8') as file:
                data = file.readlines()
        summary_start_flag = False
        for index in range(len(data)):
            if data[index] == '\n': continue
            if data[index][0] == '@':
                summary_start_flag = True
                continue
            if summary_start_flag:
                sample_summary += data[index][0:-1] + ' '
            else:
                sample_article.append(data[index][0:-1])

        candidate_summary = []
        for indexX in range(len(sample_article)):
            for indexY in range(indexX + 1, len(sample_article)):
                candidate_summary.append(' '.join([sample_article[indexX], sample_article[indexY]]))

        ########################################

        predict_result = json.load(open(os.path.join(load_path, sample_filename.replace('story', 'json')), 'r'))
        total_accuracy_counter = []
        for indexX in range(len(predict_result['predict'])):
            accuracy_counter = [1 if predict_result['predict'][indexX][indexY] ==
                                     predict_result['label'][indexX][indexY] else 0 for indexY in
                                range(len(predict_result['predict'][indexX]))]
            accuracy_counter = numpy.sum(accuracy_counter)
            total_accuracy_counter.append(accuracy_counter)

        # print(candidate_summary[numpy.argmax(total_accuracy_counter)])
        potential_predict = []
        for index in range(len(total_accuracy_counter)):
            if total_accuracy_counter[index] == numpy.max(total_accuracy_counter):
                potential_predict.append(candidate_summary[index])

        sample_score = []
        for sample in potential_predict:
            score = scorer.score(target=sample_summary, prediction=sample)
            sample_score.append([score['rouge1'].fmeasure, score['rouge2'].fmeasure, score['rougeL'].fmeasure])
        # for sample in sample_score:
        #     print(sample)
        # exit()
        total_score_max.append(sample_score[numpy.argmax([_[0] for _ in sample_score])])
        total_score_min.append(sample_score[numpy.argmin([_[0] for _ in sample_score])])
        # print(total_score)
        # exit()
    print(numpy.average(total_score_max, axis=0))
    print(numpy.average(total_score_min, axis=0))
