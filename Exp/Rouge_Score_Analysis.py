import os
import json
import tqdm
import numpy
from rouge_score import rouge_scorer

# from transformers import BartTokenizer
# [0.33838833 0.11210007 0.22704265]

if __name__ == '__main__':
    load_path = 'C:/PythonProject/DataSource-CNNDM-BART-Predict/test_delete_2/'

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_score = []
    # tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-cnn-neo')
    for filename in tqdm.tqdm(os.listdir(load_path)):
        current_sample = json.load(open(os.path.join(load_path, filename), 'r'))
        score = scorer.score(target=current_sample['summary'], prediction=current_sample['predict'])
        total_score.append([score['rouge1'].fmeasure, score['rouge2'].fmeasure, score['rougeL'].fmeasure])
    print(numpy.average(total_score, axis=0))
    final_score = numpy.average(total_score, axis=0)
# final_score = numpy.array(final_score)
# print(final_score)
# plt.plot(final_score[:, 0], label='Rouge-1')
# plt.plot(final_score[:, 1], label='Rouge-2')
# plt.plot(final_score[:, 2], label='Rouge-L')
# plt.legend()
# plt.show()
