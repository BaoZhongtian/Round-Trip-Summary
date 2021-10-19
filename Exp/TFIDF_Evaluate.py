import os
import json
import tqdm
import pickle
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

top_n = 10

if __name__ == '__main__':
    vectorizer = pickle.load(open('TFIDF_CountVectorizer.pkl', 'rb'))
    transformer = pickle.load(open('TFIDF_TfidfTransformer.pkl', 'rb'))
    reverse_vocabulary = {}
    for key in vectorizer.vocabulary_:
        reverse_vocabulary[vectorizer.vocabulary_[key]] = key

    load_path = 'C:/PythonProject/DataSource/'
    for part in ['train', 'val', 'test']:
        treat_data = json.load(open(os.path.join(load_path, 'CNNDM_%s.json' % part), 'r'))
        summary = [_['summary'] for _ in treat_data]

        predict_source = vectorizer.transform(summary)
        weight = transformer.transform(predict_source).toarray()
        total_result = []
        for sample_index in tqdm.trange(len(weight)):
            result = {'filename': treat_data[sample_index]['filename'], 'words': []}
            for top_index in range(top_n):
                result['words'].append(
                    [reverse_vocabulary[numpy.argmax(weight[sample_index])], numpy.max(weight[sample_index])])
                weight[sample_index][numpy.argmax(weight[sample_index])] = -9999
            total_result.append(result)
        # print(total_result[0])
        json.dump(total_result, open('CNNDM_SalientWords_%s.json' % part, 'w'))
