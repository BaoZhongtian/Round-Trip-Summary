import os
import json
import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    load_path = 'C:/PythonProject/DataSource/'
    test_data = json.load(open(os.path.join(load_path, 'CNNDM_test_part_shuffle.json'), 'r'))
    for batch_index in range(999, 584999, 1000):
        save_path = 'E:/ProjectData/Bart-Large-CNN-RoundTrip-0.5-ALL/Result-%08d-Test-Shuffled/' % batch_index
        model = BartForConditionalGeneration.from_pretrained(
            'E:/ProjectData/Bart-Large-CNN-RoundTrip-0.5-ALL/%08d-Encoder' % batch_index)
        # save_path = 'C:/PythonProject/DataSource-CNNDM-BART-Predict/test_shuffled'
        # model = BartForConditionalGeneration.from_pretrained(
        #     'C:/PythonProject/bart-large-cnn')

        if not os.path.exists(save_path): os.makedirs(save_path)
        tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large-cnn')
        model.eval()
        model.cuda()

        for treat_index, treat_sample in enumerate(tqdm.tqdm(test_data)):
            if os.path.exists(os.path.join(save_path, '%08d.json' % treat_index)): continue
            with open(os.path.join(save_path, '%08d.json' % treat_index), 'w'):
                pass

            treat_article = ' ' + treat_sample['article']
            inputs = tokenizer([treat_article], max_length=1000, return_tensors='pt')
            summary_ids = model.generate(inputs['input_ids'].cuda())
            summary_ids = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

            reconstruct_predict = tokenizer.decode(tokenizer.encode(' ' + treat_sample['summary']),
                                                   skip_special_tokens=True, clean_up_tokenization_spaces=False)
            current_predict = {'filename': treat_sample['filename'], 'predict': summary_ids,
                               'summary': reconstruct_predict}
            json.dump(current_predict, open(os.path.join(save_path, '%08d.json' % treat_index), 'w'))
