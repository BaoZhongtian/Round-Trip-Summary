import os
import json
import tqdm
import numpy
from Loader import CollateClass
from transformers import BartTokenizer, BartForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

batch_size = 6
cnn_path = 'C:/ProjectData/CNNDM_Dataset/cnn_stories/cnn/stories/'
dm_path = 'C:/ProjectData/CNNDM_Dataset/dailymail_stories/dailymail/stories'
save_path = 'E:/ProjectData/CNNDM_SMLM_Result/'
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    gsmlm_tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-base-gsmlm')
    gsmlm_model = BartForCausalLM.from_pretrained('C:/PythonProject/bart-base-gsmlm')
    gsmlm_model.cuda()
    gsmlm_model.eval()
    collate = CollateClass(gsmlm_tokenizer, keywords_name='SalientWords')

    test_data = json.load(open('C:/PythonProject/DataSource/CNNDM_test.json', 'r'))
    test_filename = [_['filename'] + '.story' for _ in test_data]
    for sample_filename in tqdm.tqdm(test_filename):
        if os.path.exists(os.path.join(save_path, sample_filename.replace('story', 'json'))): continue
        with open(os.path.join(save_path, sample_filename.replace('story', 'json')), 'w') as file:
            pass

        sample_article, sample_summary = [], ''
        if os.path.exists(os.path.join(cnn_path, sample_filename)):
            with open(os.path.join(cnn_path, sample_filename), 'r', encoding='UTF-8') as file:
                data = file.readlines()
        else:
            with open(os.path.join(dm_path, sample_filename), 'r', encoding='UTF-8') as file:
                data = file.readlines()
        summary_start_flag = False
        for index in range(len(data)):
            if data[index] == '\n': continue
            if data[index][0] == '@':
                summary_start_flag = True
                continue
            if summary_start_flag:
                sample_summary += data[index][0:-1] + ' '
            else:
                sample_article.append(data[index][0:-1])

        candidate_summary = []
        for indexX in range(len(sample_article)):
            for indexY in range(indexX + 1, len(sample_article)):
                candidate_summary.append(' '.join([sample_article[indexX], sample_article[indexY]]))
                # for indexZ in range(indexY + 1, len(sample_article)):
                #     candidate_summary.append(
                #         ' '.join([sample_article[indexX], sample_article[indexY], sample_article[indexZ]]))

        total_predict, total_label = [], []
        for index in range(0, len(candidate_summary), batch_size):
            treat_candidate_summary = candidate_summary[index:index + batch_size]
            batch_bundle = []
            for sample in treat_candidate_summary:
                batch_bundle.append({'filename': sample_filename.replace('.story', ''), 'summary': ' ' + sample,
                                     'article': ' ' + ' '.join(sample_article)})

            collate_result = collate.collate_keywords(batch_bundle)

            predict_result = gsmlm_model(collate_result['input_ids'].cuda())[0].argmax(dim=-1).detach().cpu().numpy()
            mlm_label = collate_result['mlm_label'].numpy()

            for indexX in range(numpy.shape(mlm_label)[0]):
                current_predict, current_label = [], []
                for indexY in range(numpy.shape(mlm_label)[1]):
                    if mlm_label[indexX][indexY] != -100:
                        current_predict.append(int(predict_result[indexX][indexY]))
                        current_label.append(int(mlm_label[indexX][indexY]))
                total_predict.append(current_predict)
                total_label.append(current_label)

        json.dump({'predict': total_predict, 'label': total_label},
                  open(os.path.join(save_path, sample_filename.replace('story', 'json')), 'w'))
