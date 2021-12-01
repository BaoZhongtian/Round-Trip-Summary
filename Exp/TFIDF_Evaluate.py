import os
import json
import tqdm
import pickle
import numpy
from transformers import BartTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

top_n = 100

if __name__ == '__main__':
    vectorizer = pickle.load(open('TFIDF_CountVectorizer.pkl', 'rb'))
    transformer = pickle.load(open('TFIDF_TfidfTransformer.pkl', 'rb'))
    reverse_vocabulary = {}
    for key in vectorizer.vocabulary_:
        reverse_vocabulary[vectorizer.vocabulary_[key]] = key

    load_path = 'C:/PythonProject/DataSource/'
    tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large-cnn')
    for part in ['Train']:
        treat_data = json.load(open(os.path.join(load_path, 'CNNDM_%s.json' % part), 'r'))

        article = []
        for treat_index in range(287227):
            with open(os.path.join(load_path, part + 'Truncation', '%08d.txt' % treat_index), 'r',
                      encoding='UTF-8') as file:
                data = file.read()
            article.append(data)

        predict_source = vectorizer.transform(article)
        weight = transformer.transform(predict_source).toarray()
        # print(predict_source)
        # print(weight)
        # exit()
        total_result = []
        for sample_index in tqdm.trange(len(weight)):
            result = {'filename': treat_data[sample_index]['filename'], 'words': []}
            for top_index in range(top_n):
                result['words'].append(
                    [reverse_vocabulary[numpy.argmax(weight[sample_index])], numpy.max(weight[sample_index])])
                weight[sample_index][numpy.argmax(weight[sample_index])] = -9999
            total_result.append(result)
        # print(total_result[0])
        json.dump(total_result, open('CNNDM_SalientWords_%s_Top100.json' % part, 'w'))
