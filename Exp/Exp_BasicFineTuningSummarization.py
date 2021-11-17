import os
import numpy
import torch
from Loader import loader_cnndm
from Tools import ProgressBar, save_model
from transformers import BartTokenizer, BartForConditionalGeneration, BartForCausalLM

episode_number = 10
learning_rate = 1E-5
save_path = 'E:/ProjectData/Bart-Large-CNN-Fine-Tuning-Restart/'
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    summary_tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large-cnn')
    summary_model = BartForConditionalGeneration.from_pretrained('C:/PythonProject/bart-large-cnn')
    if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        summary_model = torch.nn.DataParallel(summary_model, device_ids=range(torch.cuda.device_count()))
    summary_model.cuda()
    optimizer = torch.optim.RMSprop(summary_model.parameters(), lr=learning_rate)

    train_data, val_data, test_data = loader_cnndm(batch_size=3, tokenizer=summary_tokenizer, limit_size=10000)
    pbar = ProgressBar(n_total=episode_number * int(len(train_data)))

    episode_loss = 0.0
    for episode_index in range(episode_number):
        for batch_index, [batch_article, batch_summary] in enumerate(train_data):
            result = summary_model.forward(input_ids=batch_article['input_ids'].cuda(),
                                           attention_mask=batch_article['attention_mask'].cuda(),
                                           labels=batch_summary['input_ids'].cuda())
            loss = result[0]
            if torch.cuda.device_count() > 1: loss = torch.mean(loss)
            pbar(episode_index * len(train_data) + batch_index, {'loss': loss.item()})
            loss.backward()
            optimizer.step()
            summary_model.zero_grad()

            episode_loss += loss.item()
            if (episode_index * len(train_data) + batch_index) % 1000 == 999:
                print('\nTotal 1000 Loss =', episode_loss)
                episode_loss = 0.0
                save_model(summary_model.module,
                           save_path + '%08d-Parameter/' % (episode_index * len(train_data) + batch_index))
            # exit()
