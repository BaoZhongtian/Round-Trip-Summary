import os
import torch
from transformers import BartTokenizer, BartForCausalLM
from Loader import loader_cnndm
from Tools import ProgressBar, save_model

episode_number = 10
learning_rate = 1E-5
save_path = 'GSMLM-Bart/'
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained('D:/ProjectData/bart-large')
    train_loader, val_loader, test_loader = loader_cnndm(
        batch_size=4, tokenizer=tokenizer, keywords_name='SalientWords')
    model = BartForCausalLM.from_pretrained('D:/ProjectData/bart-large')

    if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    pbar = ProgressBar(n_total=episode_number * int(len(train_loader)))
    episode_loss = 0.0
    for episode_index in range(episode_number):
        for batch_index, batch_article in enumerate(train_loader):
            # article_ids, article_label = batch_article['input_ids'].cuda(), batch_article['mlm_label'].cuda()
            article_ids, article_label = batch_article['input_ids'], batch_article['mlm_label']
            loss = model(input_ids=article_ids, labels=article_label)[0]

            if torch.cuda.device_count() > 1: loss = torch.mean(loss)
            pbar(episode_index * len(train_loader) + batch_index, {'loss': loss.item()})
            loss.backward()
            optimizer.step()
            model.zero_grad()

            episode_loss += loss.item()
            if (episode_index * len(train_loader) + batch_index) % 1000 == 999:
                print('\nTotal 1000 Loss =', episode_loss)
                episode_loss = 0.0
                save_model(model.module,
                           save_path + '%08d-Encoder/' % (episode_index * len(train_loader) + batch_index))
                torch.save(
                    {'epoch': episode_index, 'optimizer': optimizer.state_dict()},
                    os.path.join(save_path, '%08d-Optimizer.pkl' % (episode_index * len(train_loader) + batch_index)))
