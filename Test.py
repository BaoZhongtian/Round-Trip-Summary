import os
import json
import tqdm
import numpy
import torch
from transformers import BartTokenizer, AutoModel

LOAD_PATH = 'C:/PythonProject/DataSource/'
batch_size = 3
CNN_PATH = 'C:/ProjectData/CNNDM_Dataset/cnn_stories/cnn/stories'
DM_PATH = 'C:/ProjectData/CNNDM_Dataset/dailymail_stories/dailymail/stories'

if __name__ == '__main__':
    test_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_test.json'), 'r'))
    keywords_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_SalientWords_Test_Top100.json'), 'r'))
    tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-base')
    mask_tokens = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
    model = AutoModel.from_pretrained('C:/PythonProject/bart-base')
    if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    for treat_index, treat_sample in enumerate(tqdm.tqdm(test_data)):
        if os.path.exists(os.path.join(CNN_PATH, treat_sample['filename'] + '.story')):
            with open(os.path.join(CNN_PATH, treat_sample['filename'] + '.story'), 'r', encoding='UTF-8') as file:
                cnn_dm_data = file.readlines()
        else:
            with open(os.path.join(DM_PATH, treat_sample['filename'] + '.story'), 'r', encoding='UTF-8') as file:
                cnn_dm_data = file.readlines()

        article_sentence, article_sentence_tokens = [], []
        for sample in cnn_dm_data:
            if sample == '\n': continue
            if sample[0] == '@': break
            article_sentence.append(sample.replace('\n', ''))
            article_sentence_tokens.append(tokenizer.encode(' ' + sample, add_special_tokens=False))
            if numpy.sum([len(_) for _ in article_sentence_tokens]) > 1000 - 2: break
        if numpy.sum([len(_) for _ in article_sentence_tokens]) > 1000 - 2:
            article_sentence = article_sentence[0:-1]
            article_sentence_tokens = article_sentence_tokens[0:-1]
        ########################################

        appoint_keywords = keywords_data[treat_index]
        assert appoint_keywords['filename'] == treat_sample['filename']
        appoint_keywords = appoint_keywords['words'][0:10]
        appoint_keywords_tokens = tokenizer.batch_encode_plus(
            [' ' + _[0] for _ in appoint_keywords], add_special_tokens=False)['input_ids']

        article_masked_tokens, article_masked_label = [], []
        for treat_sentence in article_sentence_tokens:
            current_masked_tokens, current_masked_label = [], []
            indexX = 0
            while indexX < len(treat_sentence):
                similar_flag = False
                for indexY in range(len(appoint_keywords_tokens)):
                    for indexZ in range(len(appoint_keywords_tokens[indexY])):
                        if appoint_keywords_tokens[indexY][indexZ] != treat_sentence[indexX + indexZ]: break
                    if appoint_keywords_tokens[indexY][indexZ] == treat_sentence[indexX + indexZ]:
                        similar_flag = True
                        for indexZ in range(len(appoint_keywords_tokens[indexY])):
                            current_masked_tokens.append(mask_tokens)
                            current_masked_label.append(appoint_keywords_tokens[indexY][indexZ])
                        indexX += len(appoint_keywords_tokens[indexY])
                        break
                if not similar_flag:
                    current_masked_tokens.append(treat_sentence[indexX])
                    current_masked_label.append(-100)
                    indexX += 1

            article_masked_tokens.append(current_masked_tokens)
            article_masked_label.append(current_masked_label)
        ########################################

        print('Keywords Treat Completed')

        input_token, input_lm_label, select_ids = [], [], []
        for indexX in range(len(article_sentence)):
            for indexY in range(indexX + 1, len(article_sentence)):
                select_ids.append([indexX, indexY])
        for indexX in range(len(select_ids)):
            current_token, current_lm_label = [0], [-100]
            for indexY in range(len(article_sentence)):
                if indexY == select_ids[indexX][0] or indexY == select_ids[indexX][1]:
                    current_token.extend(article_sentence_tokens[indexY])
                    current_lm_label.extend(numpy.ones(len(article_sentence_tokens[indexY])) * -100)
                else:
                    current_token.extend(article_masked_tokens[indexY])
                    current_lm_label.extend(article_masked_label[indexY])
            current_token.append(2)
            current_lm_label.append(-100)
            input_token.append(current_token)
            input_lm_label.append(current_lm_label)

        for batch_start in range(0, len(input_token), batch_size):
            batch_data = input_token[batch_start:batch_start + batch_size]
            batch_label = input_lm_label[batch_start:batch_start + batch_size]
            result = model(torch.LongTensor(batch_data).cuda())
            result = result[0].argmax(dim=-1).detach().cpu().numpy()
            # print(result[0])
            # print(tokenizer.decode(result[0]))
            # exit()

            for indexX in range(len(batch_label)):
                current_predict, current_label = [], []
                for indexY in range(len(batch_label[indexX])):
                    if batch_label[indexX][indexY] == -100: continue
                    current_predict.append(result[indexX][indexY])
                    current_label.append(batch_label[indexX][indexY])
                print(current_predict)
                print(current_label)
                exit()
            exit()
        exit()
