import os
import json
import tqdm
from transformers import BertTokenizer

if __name__ == '__main__':
    load_path = 'C:/PythonProject/DataSource/'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data = json.load(open(os.path.join(load_path, 'CNNDM_train.json'), 'r'))
    tfidf = json.load(open('Result.json', 'r'))

    train_sentence = tokenizer.encode(train_data[0]['article'], max_length=384)
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(train_sentence)))
    # exit()
    print(tfidf[0])
    # print(train_data[0]['article'])
    # print(train_data[0]['summary'])
