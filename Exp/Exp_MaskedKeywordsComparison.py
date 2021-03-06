import os
import json
import tqdm
import numpy
from Loader import CollateClass
from transformers import BartTokenizer, BartForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    load_path = 'C:/PythonProject/DataSource/'
    test_data = json.load(open(os.path.join(load_path, 'CNNDM_test.json'), 'r'))

    tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large-cnn')
    collate = CollateClass(tokenizer=tokenizer, keywords_name='SalientWords')

    model = BartForConditionalGeneration.from_pretrained(
        'C:/PythonProject/bart-large-cnn')
    model.eval()
    model.cuda()

    for masked_counter in range(1,30):
        save_path = 'C:/PythonProject/DataSource-CNNDM-BART-Predict/test_delete_%d' % masked_counter
        if not os.path.exists(save_path): os.makedirs(save_path)

        for treat_index, treat_sample in enumerate(tqdm.tqdm(test_data)):
            if os.path.exists(os.path.join(save_path, '%08d.json' % treat_index)): continue
            with open(os.path.join(save_path, '%08d.json' % treat_index), 'w'):
                pass

            inputs = collate.collate_masked_summary([treat_sample], masked_counter)
            # print(numpy.shape(inputs['input_ids']))
            # print(inputs['input_ids'])
            # print(tokenizer.decode(inputs['input_ids'][0]))
            # exit()
            summary_ids = model.generate(inputs['input_ids'].cuda())
            summary_ids = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

            reconstruct_predict = tokenizer.decode(tokenizer.encode(' ' + treat_sample['summary']),
                                                   skip_special_tokens=True, clean_up_tokenization_spaces=False)
            current_predict = {'filename': treat_sample['filename'], 'predict': summary_ids,
                               'summary': reconstruct_predict}
            json.dump(current_predict, open(os.path.join(save_path, '%08d.json' % treat_index), 'w'))
