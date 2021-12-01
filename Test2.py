import os
import json
import tqdm
import numpy
from Loader import CollateClass
from transformers import BartTokenizer, BartForConditionalGeneration

if __name__ == '__main__':
    load_path = 'C:/PythonProject/DataSource/'
    save_path = 'C:/PythonProject/DataSource/ValTruncation'
    train_data = json.load(open(os.path.join(load_path, 'CNNDM_val.json'), 'r'))
    tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large-cnn')

    for treat_index, treat_sample in enumerate(tqdm.tqdm(train_data)):
        if os.path.exists(os.path.join(save_path, '%08d.txt' % treat_index)): continue
        with open(os.path.join(save_path, '%08d.txt' % treat_index), 'w', encoding='UTF-8') as file:
            file.write(tokenizer.decode(tokenizer.encode(treat_sample['article'], max_length=1000, truncation=True),
                                        skip_special_tokens=True))
