import os
import torch
import numpy
from Loader import loader_cnndm
from transformers import EncoderDecoderModel
from Tools import ProgressBar, save_model

episode_number = 10
learning_rate = 1E-5
save_path = 'E:/ProjectData/BasicSeq2Seq-Another/'
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    train_data, val_data, test_data = loader_cnndm(batch_size=15)

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased')
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    #     save_path + '%08d-Encoder/' % 0, save_path + '%08d-Decoder/' % 0)

    if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    pbar = ProgressBar(n_total=episode_number * int(len(train_data)))

    episode_loss = 0.0
    for episode_index in range(episode_number):
        for batch_index, [batch_article, batch_summary] in enumerate(train_data):
            article_ids, article_mask = batch_article['input_ids'].cuda(), batch_article['attention_mask'].cuda()
            summary_ids, summary_mask = batch_summary['input_ids'].cuda(), batch_summary['attention_mask'].cuda()
            summary_label_masked = batch_summary['input_ids'].numpy()
            summary_label_masked = [[summary_label_masked[i][j] if summary_label_masked[i][j] > 0 else -100 for j in
                                     range(len(summary_label_masked[i]))] for i in range(len(summary_label_masked))]
            summary_label_masked = torch.LongTensor(numpy.array(summary_label_masked)).cuda()
            loss = model(input_ids=article_ids, attention_mask=article_mask, decoder_input_ids=summary_ids,
                         lm_labels=summary_label_masked)[0]
            # loss = model(input_ids=article_ids, attention_mask=article_mask, decoder_input_ids=summary_ids,
            #              decoder_attention_mask=summary_mask, lm_labels=summary_label_masked)[0]

            if torch.cuda.device_count() > 1: loss = torch.mean(loss)
            pbar(episode_index * len(train_data) + batch_index, {'loss': loss.item()})
            loss.backward()
            optimizer.step()
            model.zero_grad()

            episode_loss += loss.item()
            if (episode_index * len(train_data) + batch_index) % 1000 == 999:
                print('\nTotal 1000 Loss =', episode_loss)
                episode_loss = 0.0
                save_model(model.module.encoder,
                           save_path + '%08d-Encoder/' % (episode_index * len(train_data) + batch_index))
                save_model(model.module.decoder,
                           save_path + '%08d-Decoder/' % (episode_index * len(train_data) + batch_index))
                torch.save(
                    {'epoch': episode_index, 'optimizer': optimizer.state_dict()},
                    os.path.join(save_path, '%08d-Optimizer.pkl' % (episode_index * len(train_data) + batch_index)))
                # exit()
