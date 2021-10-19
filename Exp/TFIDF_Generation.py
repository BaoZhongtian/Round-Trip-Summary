import os
import json
import tqdm
import numpy
import pickle
from transformers import BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

top_n = 10
MAX_ARTICLE_LENGTH = 384

if __name__ == '__main__':
    load_path = 'C:/PythonProject/DataSource/'
    train_data = json.load(open(os.path.join(load_path, 'CNNDM_train.json'), 'r'))
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # corpus = [_['article'] for _ in train_data]
    summary = [_['summary'] for _ in train_data]
    # print(len(corpus))

    # batch_size = 32
    # total_article = []
    # for batch_index in tqdm.trange(0, len(corpus), batch_size):
    #     batch_data = corpus[batch_index:batch_index + batch_size]
    #     token_result = tokenizer.batch_encode_plus(batch_data, max_length=MAX_ARTICLE_LENGTH)
    #
    #     text_back = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(_)) for _ in
    #                  token_result['input_ids']]
    #     total_article.extend(text_back)
    load_path_current = 'C:/PythonProject/DataSource/Current'
    corpus = []
    for index in tqdm.trange(1000):
        part_data = json.load(open(os.path.join(load_path_current, '%08d.json' % index), 'r'))
        corpus.extend(part_data)

    vectorizer = CountVectorizer()
    article_source = vectorizer.fit_transform(corpus)
    predict_source = vectorizer.transform(summary)
    reverse_vocabulary = {}
    for key in vectorizer.vocabulary_:
        reverse_vocabulary[vectorizer.vocabulary_[key]] = key

    # 类调用
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(article_source)
    weight = transformer.transform(predict_source).toarray()
    print(numpy.shape(weight))
    pickle.dump(vectorizer, open('TFIDF_CountVectorizer.pkl', 'wb'))
    pickle.dump(transformer, open('TFIDF_TfidfTransformer.pkl', 'wb'))
    # exit()
    
    total_result = []
    for sample_index in tqdm.trange(len(weight)):
        result = {'filename': train_data[sample_index]['filename'], 'words': []}
        for top_index in range(top_n):
            result['words'].append(
                [reverse_vocabulary[numpy.argmax(weight[sample_index])], numpy.max(weight[sample_index])])
            weight[sample_index][numpy.argmax(weight[sample_index])] = -9999
        total_result.append(result)
    json.dump(total_result, open('CNNDM_SalientWords_train.json', 'w'))
