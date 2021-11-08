import os
import json
import tqdm
import numpy
from transformers import BartTokenizer, BartForConditionalGeneration

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    load_path = 'D:/ProjectData/CNNDM/'

    test_data = json.load(open(os.path.join(load_path, 'CNNDM_test.json'), 'r'))
    model = BartForConditionalGeneration.from_pretrained('D:/ProjectData/bart-large')
    tokenizer = BartTokenizer.from_pretrained('D:/ProjectData/bart-large')
    model.eval()
    # model.cuda()

    for treat_index, treat_sample in enumerate(tqdm.tqdm(test_data)):
        treat_article = treat_sample['article']
        inputs = tokenizer([treat_article], max_length=1024, return_tensors='pt')
        print(tokenizer.decode(model.generate(inputs['input_ids'], num_beams=4, max_length=128)[0]))
        exit()
        result = model.forward(inputs['input_ids'], labels=inputs['input_ids'])
        for sample in result:
            print(sample)
        exit()
        summary_ids = model.generate(inputs['input_ids'].cuda(), num_beams=16, early_stopping=True)
        summary_ids = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        reconstruct_predict = tokenizer.decode(tokenizer.encode(treat_sample['summary']), skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        print(reconstruct_predict)
        exit()
