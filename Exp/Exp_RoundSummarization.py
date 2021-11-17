import os
import torch
import numpy
from Loader import loader_cnndm
from Tools import ProgressBar, save_model
from transformers import BartTokenizer, BartForConditionalGeneration, BartForCausalLM

episode_number = 10
learning_rate = 1E-5
save_path = 'E:/ProjectData/Bart-Large-CNN-RoundTrip-0.5/'
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    gsmlm_tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-base-gsmlm')
    gsmlm_model = BartForCausalLM.from_pretrained('C:/PythonProject/bart-base-gsmlm')
    gsmlm_model.eval()
    gsmlm_model.cuda(2)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    summary_tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large-cnn')
    summary_model = BartForConditionalGeneration.from_pretrained('C:/PythonProject/bart-large-cnn')
    if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        summary_model = torch.nn.DataParallel(summary_model, device_ids=range(torch.cuda.device_count()))
    summary_model.cuda()
    optimizer = torch.optim.RMSprop(summary_model.parameters(), lr=learning_rate)

    train_loader, val_loader, test_loader = loader_cnndm(
        batch_size=2, tokenizer=summary_tokenizer, small_data_flag=True, keywords_name='SalientWords',
        keywords_masked_article_flag=True)
    print('GSMLM Part Load Completed')

    pbar = ProgressBar(n_total=episode_number * int(len(train_loader)))

    episode_loss = 0.0
    for episode_index in range(episode_number):
        for batch_index, [batch_article, batch_summary, batch_masked] in enumerate(train_loader):
            result = summary_model.forward(input_ids=batch_article['input_ids'].cuda(),
                                           attention_mask=batch_article['attention_mask'].cuda(),
                                           labels=batch_summary['input_ids'].cuda())
            loss, logits = result[0], result[1]
            if torch.cuda.device_count() > 1: loss = torch.mean(loss)

            logits = torch.argmax(logits, dim=-1)

            gsmlm_input = torch.cat([logits.cpu(), batch_masked['input_ids']], dim=1)
            gsmlm_label = torch.cat(
                [torch.LongTensor(-100 * numpy.ones(logits.size())), batch_masked['mlm_label']], dim=1)

            gsmlm_result = gsmlm_model(input_ids=gsmlm_input.cuda(2), labels=gsmlm_label.cuda(2))
            pbar(episode_index * len(train_loader) + batch_index,
                 {'summary loss': loss.item(), 'gsmlm loss': gsmlm_result[0].item()})
            loss = loss + 0.5 * gsmlm_result[0].cuda(0)
            loss.backward()
            optimizer.step()
            summary_model.zero_grad()

            episode_loss += loss.item()
            if (episode_index * len(train_loader) + batch_index) % 1000 == 999:
                print('\nTotal 1000 Loss =', episode_loss)
                episode_loss = 0.0
                save_model(summary_model.module,
                           save_path + '%08d-Encoder/' % (episode_index * len(train_loader) + batch_index))
                torch.save(
                    {'epoch': episode_index, 'optimizer': optimizer.state_dict()},
                    os.path.join(save_path, '%08d-Optimizer.pkl' % (episode_index * len(train_loader) + batch_index)))
