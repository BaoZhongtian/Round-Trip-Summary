import os
import tqdm
import json
from transformers import EncoderDecoderModel, BertTokenizer
from HistoricalRecord.Loader import loader_cnndm

load_path = 'E:/ProjectData/BasicSeq2Seq-Another/'
save_path = 'E:/ProjectData/BasicSeq2Seq-Another-Evaluate/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    train_data, val_data, test_data = loader_cnndm(batch_size=8)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for evaluate_index in range(160, 0, -1):
        fold_name = '%08d' % (evaluate_index * 1000 + 999)
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            os.path.join(load_path, fold_name + '-Encoder'), os.path.join(load_path, fold_name + '-Decoder'))
        # if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.cuda()

        total_generated = []
        for batch_index, [batch_article, batch_summary] in enumerate(tqdm.tqdm(train_data)):
            article_ids, article_mask = batch_article['input_ids'].cuda(), batch_article['attention_mask'].cuda()
            generated = model.generate(input_ids=article_ids, num_beams=5, max_length=128, early_stopping=True,
                                       dercoder_start_token_id=model.config.decoder.pad_token_id,
                                       bos_token_id=model.config.decoder.pad_token_id)
            generated = generated.detach().cpu().numpy().tolist()
            total_generated.extend(generated)

        json.dump(total_generated, open(os.path.join(save_path, '%03d-Train.json' % evaluate_index), 'w'))
        # exit()
