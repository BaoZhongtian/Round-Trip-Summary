import os
import json
import tqdm
from transformers import BertTokenizer

MAX_ARTICLE_LENGTH = 384

if __name__ == '__main__':
    load_path = 'C:/PythonProject/DataSource/'
    save_path = 'C:/PythonProject/DataSource/Current/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    train_data = json.load(open(os.path.join(load_path, 'CNNDM_train.json'), 'r'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    corpus = [_['article'] for _ in train_data]
    summary = [_['summary'] for _ in train_data]
    print(len(corpus))

    batch_size = 32
    total_article = []
    for batch_index in tqdm.trange(0, len(corpus), batch_size):
        if os.path.exists(os.path.join(save_path, '%08d.json' % int(batch_index / batch_size))): continue
        with open(os.path.join(save_path, '%08d.json' % int(batch_index / batch_size)), 'w'):
            pass

        batch_data = corpus[batch_index:batch_index + batch_size]
        token_result = tokenizer.batch_encode_plus(batch_data, max_length=MAX_ARTICLE_LENGTH)

        text_back = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(_)) for _ in
                     token_result['input_ids']]
        json.dump(text_back, open(os.path.join(save_path, '%08d.json' % int(batch_index / batch_size)), 'w'))
