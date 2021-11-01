import os
import json
import tqdm
import numpy
from rouge_score import rouge_scorer

if __name__ == '__main__':
    load_path = 'C:/PythonProject/DataSource-CNNDM-BART-Predict/val/'
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_score = []
    for filename in tqdm.tqdm(os.listdir(load_path)):
        current_sample = json.load(open(os.path.join(load_path, filename), 'r'))
        score = scorer.score(target=current_sample['summary'], prediction=current_sample['predict'])
        total_score.append([score['rouge1'].fmeasure, score['rouge2'].fmeasure, score['rougeL'].fmeasure])
    print(numpy.average(total_score, axis=0))
