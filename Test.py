import os
import json
import tqdm
import numpy
from rouge_score import rouge_scorer

# from transformers import BartTokenizer

if __name__ == '__main__':
    final_score = []
    for batch_index in range(999, 49999, 1000):
        load_path = 'E:/ProjectData/Bart-Large-CNN-RoundTrip/Result-%08d-Test-Shuffled/' % batch_index
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        total_score = []
        # tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-cnn-neo')
        for filename in tqdm.tqdm(os.listdir(load_path)[0:1000]):
            current_sample = json.load(open(os.path.join(load_path, filename), 'r'))
            score = scorer.score(target=current_sample['summary'], prediction=current_sample['predict'])
            total_score.append([score['rouge1'].fmeasure, score['rouge2'].fmeasure, score['rougeL'].fmeasure])
        print(numpy.average(total_score, axis=0))
        final_score.append(numpy.average(total_score, axis=0))
    final_score = numpy.array(final_score)
    print(final_score[numpy.argmax(numpy.array(final_score[:, 0]))])
    print(numpy.argmax(final_score[:, 0]))
    # print(final_score)
