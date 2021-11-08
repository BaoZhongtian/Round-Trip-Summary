from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('D:/ProjectData/bart-cnn-neo')
mask_id = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
print(mask_id)